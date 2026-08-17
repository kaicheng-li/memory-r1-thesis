import os
import sys
import math
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import re
import warnings
import torch
import torch.distributed as dist
import torch.nn.functional as F
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoModel,AutoTokenizer, AutoModelForCausalLM
from lm_dataset import RLAIFDataset
from trainer_utils import Logger, is_main_process, lm_checkpoint, setup_seed, SkipBatchSampler,init_distributed_mode

warnings.filterwarnings('ignore')

class HFCritic(nn.Module):
    def __init__(self, model_name, torch_dtype=torch.bfloat16):
        super().__init__()
        self.backbone = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        ) 
        #self.backbone作用是特征提取，常用网络：ResNet, VGG, MobileNet,
        #在torch中常用于迁移学习：backbone 通常会在大型数据集（如 ImageNet）上进行预训练。你可以在自己的任务中直接加载这个训练好的 backbone，这能极大地加快训练速度并提升模型性能。
        #或者可用于模块化设计：将 backbone 作为一个独立的模块，连接到不同的头部（如分类头、回归头等），以适应不同的任务需求。
        hidden_size = self.backbone.config.hidden_size
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask=None):
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True, #告诉模型不仅要输出最终的计算结果，还要把每一层 Transformer 的输出（隐藏状态）都保存下来。
            use_cache=False,  #通常在训练时设置为 False，在推理（生成文本）时设置为 True 以加速。这里关闭缓存，意味着不复用之前的计算结果
        )
        #这里的out输出的是每一层 Transformer 的输出（隐藏状态）
        h = out.hidden_states[-1]         # 即[B, T, H]的最后一个H，隐藏层维度换言之上下文特征
        v = self.value_head(h).squeeze(-1) # self.value_head：这是一个简单的神经网络层（通常是一个线性层 nn.Linear），它负责将高维特征 H 映射到你需要的任务空间。
        #.squeeze(-1)：value_head 的输出通常形状是 [B, T, 1]（最后一个维度是 1）。squeeze(-1) 的作用是去掉这个长度为 1 的维度，将其变成 [B, T]
        return v


def compute_gae(rewards, values, masks, gamma, lam):
    """
    广义优势估计（Generalized Advantage Estimation, GAE）
    rewards, values, masks: [B, T]
    masks[t] = 0 marks a terminal step (no value bootstrap beyond it).
    Returns token-level advantages and value targets.
    """
    B, T = rewards.shape
    advantages = torch.zeros_like(rewards) #创建一个与输入张量（Tensor）具有相同形状（shape）、数据类型（dtype）和所在设备（device）的新张量，并将其中所有元素都填充为 0
    last_gae = 0
    values_ext = torch.cat([values, torch.zeros(B, 1, device=values.device)], dim=1) #dim=1指定拼接的维度是第 1 维（也就是列，即序列长度方向）
    for t in reversed(range(T)):
        delta = rewards[:, t] + gamma * values_ext[:, t + 1] * masks[:, t] - values_ext[:, t] #计算每个时间步的 时序差分误差（TD Error），也称为 Delta,这是“实际获得的回报”与“模型预测的回报”之间的差异
        last_gae = delta + gamma * lam * masks[:, t] * last_gae #利用一个递归公式，将当前的 Delta 和未来的优势值加权平均，得到最终的优势估计 A_t,At=δt+(γλ)A_t+1
        advantages[:, t] = last_gae
    return advantages, advantages + values  # advantages, value targets (returns)

def calculate_rewards(prompts, responses, reward_model, reward_tokenizer,args):
    #专门用于强化学习,特别是训练推理模型
    #输入：提示词（prompts）、模型生成的回答（responses）、奖励模型（reward_model）等
    """整合所有奖励函数计算总奖励"""
    def reasoning_model_reward(rewards):
        # 1. 格式奖励（仅针对训练推理模型时使用）
        pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
        pattern2 = r"^<think>\n.*?\n</think>\n\n<answer>\n.*?\n</answer>$"

        matches_pattern = [re.match(pattern, response, re.S) for response in responses]
        matches_pattern2 = [re.match(pattern2, response, re.S) for response in responses]

        format_rewards = []
        for match_pattern, match_pattern2 in zip(matches_pattern, matches_pattern2):
            if match_pattern:
                format_rewards.append(0.5)
            elif match_pattern2:
                format_rewards.append(0.5)
            else:
                format_rewards.append(0.0)  
        #如果回答完全符合预期格式，奖励0.5；如果回答符合第二种格式（允许think和answer之间有额外的换行），也奖励0.5；否则奖励0.0
        rewards += torch.tensor(format_rewards, device=args.device) #将格式奖励转换为张量，并加到总奖励上

        # 2. 标记奖励（防止严格奖励稀疏，仅针对训练推理模型时使用）
        def mark_num(text):
            reward = 0
            if text.count("<think>") == 1:
                reward += 0.25
            if text.count("</think>") == 1:
                reward += 0.25
            if text.count("<answer>") == 1:
                reward += 0.25
            if text.count("</answer>") == 1:
                reward += 0.25
            return reward

        mark_rewards = [mark_num(response) for response in responses]
        rewards += torch.tensor(mark_rewards, device=args.device)
        return rewards

    rewards = torch.zeros(len(responses), device=args.device) #rewards初始化

    # 格式奖励
    if args.reasoning == 1:
        rewards = reasoning_model_reward(rewards)

    # 使用reward model计算整个response的奖励
    with torch.no_grad():
        reward_model_scores = []
        for prompt, response in zip(prompts, responses):
            pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
            matches = re.findall(pattern, prompt, re.DOTALL) #使用正则表达式匹配类似 <|im_start|>user ... <|im_end|> 的特殊标记
            messages = [{"role": role, "content": content.strip()} for role, content in matches]

            tmp_chat = messages + [{"role": "assistant", "content": response}]  #将混乱的字符串转换成结构化的对话历史列表（messages）
            score = reward_model.get_score(reward_tokenizer, tmp_chat) #将完整的对话喂给 reward_model，调用其 get_score 方法

            scale = 3.0
            score = max(min(score, scale), -scale)  #将分数强制限制在 [-3.0, 3.0] 之间。

            # 当args.reasoning=1时，额外计算<answer>内容的奖励
            if args.reasoning == 1:
                answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
                if answer_match:
                    answer_content = answer_match.group(1).strip()
                    # 对answer内容单独计算reward
                    tmp_chat = messages + [{"role": "assistant", "content": answer_content}]
                    answer_score = reward_model.get_score(reward_tokenizer, tmp_chat)
                    answer_score = max(min(answer_score, scale), -scale)
                    score = score * 0.4 + answer_score * 0.6
            reward_model_scores.append(score)

        reward_model_scores = torch.tensor(reward_model_scores, device=args.device)
        rewards += reward_model_scores

    return rewards


def ppo_train_epoch(epoch, loader, iters, old_actor_model, ref_model, actor_scheduler, critic_scheduler, reward_model, reward_tokenizer,autocast_ctx, args,tokenizer,start_step=0, wandb=None):
    """这段代码在一个数据加载器（loader）上循环，对每个批次的数据执行以下四大步骤：
    Rollout（生成）：Actor模型生成回答。
    Reward（打分）：使用奖励模型和规则给回答打分。
     GAE（计算优势）：计算优势函数，判断动作的好坏。
    Update（更新）：更新Actor（策略网络）和Critic（价值网络）。"""
            
    actor_model.train()
    critic_model.train()

    for step, batch in enumerate(loader, start=start_step + 1):
        prompts = batch["prompt"]  # list[str], length B
        # ── Phase 1: Rollout ─────────────────────────────────────────────────────────
        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, 
                       max_length=args.max_seq_len, padding_side="left").to(args.device)  # input_ids: [B, P], attention_mask: [B, P]
        prompt_length = enc.input_ids.shape[1]

        with torch.no_grad():
            # DDP 模型需要使用 .module 访问 generate 方法
            model_for_gen = actor_model.module if isinstance(actor_model, DistributedDataParallel) else actor_model
            gen_out = model_for_gen.generate(
                input_ids=enc.input_ids, attention_mask=enc.attention_mask,
                max_new_tokens=args.max_gen_len, do_sample=True, temperature=0.8,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)  # 生成回答，B是batch,P是prompt,R是response [B, P+R]
        # ── Phase 2: Rewards ───────
        responses_text = [tokenizer.decode(gen_out[i, prompt_length:], skip_special_tokens=True) for i in range(len(prompts))]  #len(prompts)实际是batch size,最终该行输出的是[B,R]
        rewards = calculate_rewards(prompts, responses_text, reward_model, reward_tokenizer,args)  # [B]
        
        # ── Phase 3: Masks，作用：让Critic模型对生成序列中每个位置的价值进行估计。 ────────────────
        full_mask = (gen_out != tokenizer.pad_token_id).long()  # [B, P+R],创建掩码 (区分有效token和padding),pad_token_id是tokenizer中用于填充的特殊token ID，full_mask中有效token位置为1，padding位置为0
        labels = gen_out[:, 1:]   # gen_out是[B,P+R]，被处理为[B, T-1]因为在语言模型训练中，通常会将输入序列向右移动一个位置来创建标签（labels）。这样，模型在预测下一个 token 时，可以看到当前 token 作为输入，而标签则是下一个 token。
        
        values_seq = critic_model(input_ids=gen_out, attention_mask=full_mask)  # crtic模型预测每个token的价值，[B, P+R]
        last_indices = (full_mask * torch.arange(full_mask.size(1), device=gen_out.device)).argmax(dim=1)  # 获取每个序列最后一个有效token的位置
        values = values_seq[torch.arange(values_seq.size(0), device=values_seq.device), last_indices]  # [B]，提取最后一个token的价值 (用于后续的bootstrap)

        # ── Phase 4: Actor forward & GAE ─────────────────────────────────────────
        """构建Token级奖励：
            将之前计算的序列级奖励（rewards）放置在序列最后一个有效token的位置上，其余位置为0。
            token_rewards[torch.arange(len(prompts)), last_indices - 1] = rewards
        计算优势 (GAE)：
            调用 compute_gae 函数（你之前问过的），计算每个token的优势值 advs 和回报 returns。
            对优势值进行标准化（减均值，除标准差），以稳定训练。
        计算对数概率 (Log Prob)：
            前向传播Actor模型，计算生成当前序列的对数概率 actor_logp。
            同时计算旧策略（old_actor_model）和参考策略（ref_model，通常是SFT模型）的对数概率。
        计算KL散度：
            kl = (actor_logp - old_logp).mean()：用于PPO的惩罚项，防止新策略偏离旧策略太远。
            kl_ref：用于监控策略相对于初始SFT模型的偏离程度。"""
        rewards = rewards * args.reward_scale
        # 仅在每条序列最后一个有效 token 放置 reward（避免 prompt 位置累积虚假奖励）
        token_rewards = torch.zeros_like(values_seq[:, :-1])
        token_rewards[torch.arange(len(prompts)), last_indices - 1] = rewards

        advs, returns = compute_gae(token_rewards, values_seq[:, :-1],
                                    full_mask[:, 1:], args.gamma, args.lam)
        advantages = advs.sum(dim=1)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        with autocast_ctx:
            logits = actor_model(input_ids=gen_out, attention_mask=full_mask).logits   # [B, T, V]
            
        logp_tokens = F.log_softmax(logits[:, :-1], dim=-1).gather(2, labels.unsqueeze(-1)).squeeze(-1)  # [B, P+R-1]
        
        seq_len = gen_out.size(1) - 1
        resp_mask = torch.arange(seq_len, device=gen_out.device).unsqueeze(0) >= prompt_length - 1
        final_mask = resp_mask & (~labels.eq(tokenizer.pad_token_id))  # [B, P+R-1]
        actor_logp = (logp_tokens * final_mask).sum(dim=1)  # [B]

        with torch.no_grad():
            old_logits = old_actor_model(input_ids=gen_out, attention_mask=full_mask).logits  # [B, P+R, V]
            old_logp_tokens = F.log_softmax(old_logits[:, :-1], dim=-1).gather(2, labels.unsqueeze(-1)).squeeze(-1)  # [B, P+R-1]
            old_logp = (old_logp_tokens * final_mask).sum(dim=1)  # [B]
            
            ref_logits = ref_model(input_ids=gen_out, attention_mask=full_mask).logits  # [B, P+R, V]
            ref_logp_tokens = F.log_softmax(ref_logits[:, :-1], dim=-1).gather(2, labels.unsqueeze(-1)).squeeze(-1)  # [B, P+R-1]
            ref_logp = (ref_logp_tokens * final_mask).sum(dim=1)  # [B]

        kl = (actor_logp - old_logp).mean()  # scalar
        kl_ref = (actor_logp - ref_logp).mean()  # scalar
        
        
        # PPO 损失计算
        ratio = torch.exp(actor_logp - old_logp)  # [B]
        surr1 = ratio * advantages  # [B]
        surr2 = torch.clamp(ratio, 1.0 - args.clip_epsilon, 1.0 + args.clip_epsilon) * advantages  # [B]
        policy_loss = -torch.min(surr1, surr2).mean()  # scalar
        # Critic 损失 (MSE)
        value_target = returns[torch.arange(len(prompts)), last_indices - 1]  # [B]
        value_loss = F.mse_loss(values, value_target.detach())  # scalar
        # 总损失 (包含KL惩罚)
        loss = (policy_loss + args.vf_coef * value_loss + args.kl_coef * kl_ref) / args.accumulation_steps  # scalar
        # 反向传播与优化
        loss.backward()
        
        # 以下为(梯度裁剪、优化器step、学习率调度)
        if (step + 1) % args.accumulation_steps == 0:  
            #梯度累积，作用：模拟更大的批次大小（Batch Size）。
            #如果显存不够，可以通过累积多个小批次的梯度，再进行一次参数更新，以达到类似大批次训练的效果
            clip_grad_norm_(actor_model.parameters(), args.max_grad_norm)
            clip_grad_norm_(critic_model.parameters(), args.max_grad_norm)
            actor_optimizer.step()
            critic_optimizer.step()
            actor_optimizer.zero_grad()
            critic_optimizer.zero_grad()
            actor_scheduler.step()
            critic_scheduler.step()
            
        if is_main_process():
            response_ids = gen_out[:, prompt_length:]
            is_eos = (response_ids == tokenizer.eos_token_id)
            eos_indices = torch.argmax(is_eos.int(), dim=1)
            has_eos = is_eos.any(dim=1)
            lengths = torch.where(has_eos, eos_indices + 1, torch.tensor(response_ids.shape[1], device=is_eos.device))
            avg_len = lengths.float().mean()

            actor_loss_val = policy_loss.item()
            critic_loss_val = value_loss.item()
            reward_val = rewards.mean().item()
            kl_val = kl.item()
            kl_ref_val = kl_ref.item()
            avg_len_val = avg_len.item()
            actor_lr = actor_optimizer.param_groups[0]['lr']
            critic_lr = critic_optimizer.param_groups[0]['lr']

            if wandb is not None:
                wandb.log({
                    "actor_loss": actor_loss_val,
                    "critic_loss": critic_loss_val,
                    "reward": reward_val,
                    "kl": kl_val,
                    "kl_ref": kl_ref_val,
                    "avg_response_len": avg_len_val,
                    "actor_lr": actor_lr,
                })

            Logger(f"Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), "
                   f"Actor Loss: {actor_loss_val:.4f}, Critic Loss: {critic_loss_val:.4f}, "
                   f"Reward: {reward_val:.4f}, KL: {kl_val:.4f}, KL_ref: {kl_ref_val:.4f}, "
                   f"Avg Response Len: {avg_len_val:.2f}, Actor LR: {actor_lr:.8f}, Critic LR: {critic_lr:.8f}")

        if (step + 1) % args.update_old_actor_freq == 0:
            raw_actor = actor_model.module if isinstance(actor_model, DistributedDataParallel) else actor_model
            raw_actor = getattr(raw_actor, '_orig_mod', raw_actor)
            state_dict = raw_actor.state_dict()
            old_actor_model.load_state_dict({k: v.detach().cpu() for k, v in state_dict.items()})
            old_actor_model.to(args.device)

        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            actor_model.eval()
            ckp = f'{args.save_dir}/{args.save_weight}.pth'
            raw_actor = actor_model.module if isinstance(actor_model, DistributedDataParallel) else actor_model
            raw_actor = getattr(raw_actor, '_orig_mod', raw_actor)
            actor_state = raw_actor.state_dict()
            torch.save({k: v.half().cpu() for k, v in actor_state.items()}, ckp)
            
            # 使用 lm_checkpoint 保存完整状态（包括 critic）
            lm_checkpoint(weight=args.save_weight, model=actor_model, optimizer=actor_optimizer, 
                         epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints',
                         scheduler=actor_scheduler, critic_model=critic_model, 
                         critic_optimizer=critic_optimizer, critic_scheduler=critic_scheduler)
            actor_model.train()
            del actor_state

        del enc, gen_out, responses_text, rewards, full_mask, values_seq, values, advantages
        del logits, labels, logp_tokens, final_mask, actor_logp, old_logits, old_logp, ref_logits, ref_logp
        del kl, kl_ref, ratio, surr1, surr2, policy_loss, value_loss, loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind PPO (Proximal Policy Optimization)")
    parser.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    parser.add_argument("--lam", type=float, default=0.95, help="GAE参数")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="基础模型名称或路径")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='ppo_actor', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=8e-8, help="Actor学习率")
    parser.add_argument("--critic_learning_rate", type=float, default=8e-8, help="Critic学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--save_interval", type=int, default=10, help="模型保存间隔")
    parser.add_argument('--max_seq_len', default=66, type=int, help="Prompt最大长度")
    parser.add_argument("--max_gen_len", type=int, default=1536, help="生成的最大长度")
    parser.add_argument("--data_path", type=str, default="../dataset/rlaif-mini.jsonl", help="RLAIF数据路径")
    parser.add_argument("--clip_epsilon", type=float, default=0.1, help="PPO裁剪参数")
    parser.add_argument("--vf_coef", type=float, default=0.5, help="Value function系数")
    parser.add_argument("--reward_scale", type=float, default=1.0, help="奖励缩放系数")
    parser.add_argument("--reasoning", type=int, default=1, choices=[0, 1], help='推理模型类型（0=普通模型，1=推理模型）')
    parser.add_argument("--update_old_actor_freq", type=int, default=4, help="更新old_actor_model的频率")
    parser.add_argument("--reward_model_path", type=str, default="../../internlm2-1_8b-reward", help="Reward模型路径")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-PPO", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="梯度裁剪的最大范数")
    parser.add_argument("--kl_coef", type=float, default=0.02, help="KL惩罚系数（相对ref模型）")
    
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    """启动并配置分布式训练环境。
    local_rank:是当前进程在本地节点（本机）上的 GPU 编号
    调用 torch.distributed.init_process_group 来让所有进程互相认识并建立通信
    torch.cuda.set_device(local_rank)是代码在哪张卡上运行"""
    local_rank = init_distributed_mode()  
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    ckp_data = lm_checkpoint(weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None
    
    # ========== 3. 设置混合精度 ==========
    """核心思想是：并非所有计算都需要高精度.
    大部分计算使用16位浮点数（低精度）,关键部分保留32位浮点数（高精度）
    关键技术：损失缩放（Loss Scaling）
    autocast作用：这是一个上下文管理器（Context Manager）。
    在 with autocast(): 代码块内，PyTorch 会自动为不同的操作选择最合适的数据类型。
    GradScaler作用：实现损失缩放（Loss Scaling）
    scaler.scale(loss)：在反向传播前，将损失值乘以一个动态调整的缩放因子，防止梯度在 FP16/BF16 下变为零。
    scaler.step(optimizer)：在更新权重前，自动将梯度“反缩放”回正常范围，并处理可能出现的梯度溢出（Inf/NaN）。
    scaler.update()：根据本次迭代的梯度情况，自动调整下一次的缩放因子。如果梯度没有溢出，它可能会增加缩放因子以提高效率；如果发生溢出，则会减少缩放因子"""
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. 配wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-PPO-Epoch-{args.epochs}-BS-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 初始化模型和数据 ==========
    # Actor模型
    actor_model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=dtype, trust_remote_code=True).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    if args.use_compile == 1: 
        actor_model = torch.compile(actor_model) #torch.compile是个优化引擎，用于推理和训练加速
        Logger('torch.compile enabled')
    # Old Actor模型
    old_actor_model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=dtype, trust_remote_code=True).to(args.device).eval().requires_grad_(False)
    # Reference模型
    ref_model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=dtype, trust_remote_code=True).to(args.device).eval().requires_grad_(False)
    # Critic模型
    model_tag = args.model_name.replace('/', '_')
    ckp = f'{args.save_dir}/{model_tag}.pth'
    state_dict = torch.load(ckp, map_location=args.device)
    critic_model = HFCritic(args.model_name, torch_dtype=dtype)
    critic_model.load_state_dict(state_dict, strict=False)
    critic_model = critic_model.to(args.device)
    # Reward模型
    reward_model = AutoModel.from_pretrained(
        args.reward_model_path, torch_dtype=torch.float16, trust_remote_code=True
    )
    reward_model = reward_model.to(args.device).eval().requires_grad_(False)
    reward_tokenizer = AutoTokenizer.from_pretrained(args.reward_model_path, trust_remote_code=True)
    # 数据和优化器
    train_ds = RLAIFDataset(args.data_path, tokenizer, max_length=(args.max_seq_len + args.max_gen_len))
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    actor_optimizer = optim.AdamW(actor_model.parameters(), lr=args.learning_rate)
    critic_optimizer = optim.AdamW(critic_model.parameters(), lr=args.critic_learning_rate)
    iters = math.ceil(len(train_ds)/args.batch_size)
    total_optimizer_steps = (iters // args.accumulation_steps) * args.epochs
    actor_scheduler = CosineAnnealingLR(actor_optimizer, T_max=total_optimizer_steps, eta_min=args.learning_rate / 10)
    critic_scheduler = CosineAnnealingLR(critic_optimizer, T_max=total_optimizer_steps, eta_min=args.critic_learning_rate / 10)
    
    # ========== 6. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        actor_model.load_state_dict(ckp_data['model'])
        critic_model.load_state_dict(ckp_data['critic_model'])
        actor_optimizer.load_state_dict(ckp_data['optimizer'])
        critic_optimizer.load_state_dict(ckp_data['critic_optimizer'])
        actor_scheduler.load_state_dict(ckp_data['scheduler'])
        critic_scheduler.load_state_dict(ckp_data['critic_scheduler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 7. DDP包模型 ==========
    if dist.is_initialized():
        actor_model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        critic_model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        actor_model = DistributedDataParallel(actor_model, device_ids=[local_rank])
        critic_model = DistributedDataParallel(critic_model, device_ids=[local_rank])
        old_actor_model.to(args.device)
    
    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0: 
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            ppo_train_epoch(epoch, loader, len(loader) + skip, old_actor_model, ref_model, 
                           actor_scheduler, critic_scheduler, reward_model, reward_tokenizer, autocast_ctx, args, tokenizer, start_step=start_step, wandb=wandb)
        else:
            ppo_train_epoch(epoch, loader, len(loader), old_actor_model, ref_model, 
                           actor_scheduler, critic_scheduler, reward_model, reward_tokenizer, autocast_ctx, args, tokenizer, start_step,wandb=wandb)
    
    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized(): dist.destroy_process_group()