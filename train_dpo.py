import os
import sys
import math
import time
import warnings
import argparse

import torch
import torch.nn.functional as F
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.utils import clip_grad_norm_
from transformers import AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────────────────────────────────────
# 日志工具
# ─────────────────────────────────────────────────────────────────────────────
def is_main_process():
    return (not dist.is_initialized()) or dist.get_rank() == 0


def Logger(content):
    if is_main_process():
        print(content)


# ─────────────────────────────────────────────────────────────────────────────
# 分布式初始化
# ─────────────────────────────────────────────────────────────────────────────
def init_distributed_mode():
    """初始化 DDP 环境，返回 local_rank（单卡时返回 0）。"""
    if "LOCAL_RANK" not in os.environ:
        return 0
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return local_rank


def setup_seed(seed):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────────────────────────────────────
# 通用 DPO 数据集
# ─────────────────────────────────────────────────────────────────────────────
class DPODataset(Dataset):
    """
    从 JSONL 文件中读取 DPO 偏好对数据。

    每行 JSON 需包含以下字段（与 HuggingFace TRL 格式兼容）：
        - "prompt"   : str  提示词
        - "chosen"   : str  优选回答
        - "rejected" : str  劣选回答

    tokenize 策略：将 prompt + chosen / prompt + rejected 分别 tokenize，
    提取 response 部分的 mask 用于 DPO Loss 计算。
    """

    def __init__(self, data_path: str, tokenizer, max_length: int = 1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        import json
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                # 兼容两种格式：
                #   1. {"prompt": ..., "chosen": ..., "rejected": ...}
                #   2. {"conversations": [{"role": "user", ...}], "chosen": ..., "rejected": ...}
                prompt = obj.get("prompt", "")
                if not prompt and "conversations" in obj:
                    conv = obj["conversations"]
                    # 取最后一条 user 消息作为 prompt
                    for turn in reversed(conv):
                        if turn.get("role") == "user":
                            prompt = turn.get("content", "")
                            break
                chosen   = obj.get("chosen", "")
                rejected = obj.get("rejected", "")
                if prompt and chosen and rejected:
                    self.samples.append((prompt, chosen, rejected))

    def __len__(self):
        return len(self.samples)

    def _encode_pair(self, prompt: str, response: str):
        """
        将 (prompt, response) 编码为模型输入，并生成仅覆盖 response 部分的 mask。
        返回 input_ids [max_length]，response_mask [max_length]。
        """
        tokenizer = self.tokenizer
        # 尝试使用 chat template；若不支持则回退到拼接字符串
        try:
            full_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt},
                 {"role": "assistant", "content": response}],
                tokenize=False,
                add_generation_prompt=False,
            )
            prompt_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            full_text   = prompt + response
            prompt_text = prompt

        full_ids   = tokenizer(full_text,   add_special_tokens=False).input_ids
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids

        prompt_len = len(prompt_ids)
        full_len   = len(full_ids)

        # 截断到 max_length
        if full_len > self.max_length:
            full_ids = full_ids[:self.max_length]
            full_len = self.max_length
        # Padding
        pad_len  = self.max_length - full_len
        pad_id   = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        input_ids = full_ids + [pad_id] * pad_len

        # response mask：只对 response token 计 loss，prompt 部分及 padding 置 0
        mask = [0] * self.max_length
        for i in range(prompt_len, full_len):
            mask[i] = 1

        return input_ids, mask

    def __getitem__(self, idx):
        prompt, chosen, rejected = self.samples[idx]

        x_chosen,   mask_chosen   = self._encode_pair(prompt, chosen)
        x_rejected, mask_rejected = self._encode_pair(prompt, rejected)

        # 标签 = input_ids 右移一位（next-token prediction）
        def make_label(ids):
            ids_t = torch.tensor(ids, dtype=torch.long)
            label = ids_t.clone()
            pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            label[label == pad_id] = -100   # 忽略 pad 位置的 loss
            return ids_t, label

        x_c, y_c = make_label(x_chosen)
        x_r, y_r = make_label(x_rejected)

        return {
            "x_chosen":    x_c,
            "y_chosen":    y_c,
            "mask_chosen": torch.tensor(mask_chosen, dtype=torch.float),
            "x_rejected":    x_r,
            "y_rejected":    y_r,
            "mask_rejected": torch.tensor(mask_rejected, dtype=torch.float),
        }


# ─────────────────────────────────────────────────────────────────────────────
# DPO 核心函数
# ─────────────────────────────────────────────────────────────────────────────
def logits_to_log_probs(logits, labels):
    """
    将模型输出 logits 转换为 per-token log 概率。
    logits: (B, T, V)   labels: (B, T)  →  log_probs: (B, T)
    """
    log_probs = F.log_softmax(logits, dim=-1)
    # 取对应 label token 的 log prob
    log_probs_per_token = torch.gather(
        log_probs, dim=2, index=labels.unsqueeze(2)
    ).squeeze(-1)
    return log_probs_per_token


def dpo_loss(ref_log_probs, policy_log_probs, mask, beta):
    """
    标准 DPO Loss（Direct Preference Optimization, Rafailov et al. 2023）：

        L_DPO = -E[ log σ( β · (log π_θ(y_w|x)/π_ref(y_w|x)
                                - log π_θ(y_l|x)/π_ref(y_l|x)) ) ]

    ref_log_probs, policy_log_probs: (2*B, T)
        前半 B 行对应 chosen，后半 B 行对应 rejected
    mask: (2*B, T)  仅 response token 处为 1
    beta: 温度系数，控制偏好偏移强度
    """
    # 对 response 区域取平均 log prob（按长度归一化）
    seq_lengths = mask.sum(dim=1, keepdim=True).clamp_min(1e-8)
    ref_log_probs    = (ref_log_probs    * mask).sum(dim=1) / seq_lengths.squeeze()
    policy_log_probs = (policy_log_probs * mask).sum(dim=1) / seq_lengths.squeeze()

    # 拆分 chosen / rejected
    B = ref_log_probs.shape[0] // 2
    chosen_ref_lp    = ref_log_probs[:B]
    rejected_ref_lp  = ref_log_probs[B:]
    chosen_pi_lp     = policy_log_probs[:B]
    rejected_pi_lp   = policy_log_probs[B:]

    # 策略和参考模型的 log-ratio 之差
    pi_logratios  = chosen_pi_lp   - rejected_pi_lp
    ref_logratios = chosen_ref_lp  - rejected_ref_lp
    logits = pi_logratios - ref_logratios          # β 缩放前的 DPO 分数

    loss = -F.logsigmoid(beta * logits)
    return loss.mean()


# ─────────────────────────────────────────────────────────────────────────────
# 跳步 BatchSampler（支持续训）
# ─────────────────────────────────────────────────────────────────────────────
class SkipBatchSampler:
    def __init__(self, sampler_or_indices, batch_size, skip_batches=0):
        self.sampler = sampler_or_indices
        self.batch_size = batch_size
        self.skip = skip_batches

    def __iter__(self):
        batch = []
        skipped = 0
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                if skipped < self.skip:
                    skipped += 1
                else:
                    yield batch
                batch = []

    def __len__(self):
        total = sum(1 for _ in self.sampler) // self.batch_size
        return max(0, total - self.skip)


# ─────────────────────────────────────────────────────────────────────────────
# 训练 Epoch
# ─────────────────────────────────────────────────────────────────────────────
def train_epoch(epoch, loader, iters, ref_model, scheduler, start_step=0, wandb=None, beta=0.1):
    start_time = time.time()
    model.train()

    for step, batch in enumerate(loader, start=start_step + 1):
        x_chosen    = batch['x_chosen'].to(args.device)
        x_rejected  = batch['x_rejected'].to(args.device)
        y_chosen    = batch['y_chosen'].to(args.device)
        y_rejected  = batch['y_rejected'].to(args.device)
        mask_chosen    = batch['mask_chosen'].to(args.device)
        mask_rejected  = batch['mask_rejected'].to(args.device)

        # 将 chosen 和 rejected 拼成一个大 batch 一次前向，节省显存
        x    = torch.cat([x_chosen,   x_rejected],   dim=0)   # (2B, T)
        y    = torch.cat([y_chosen,   y_rejected],   dim=0)   # (2B, T)
        mask = torch.cat([mask_chosen, mask_rejected], dim=0)  # (2B, T)

        with autocast_ctx:
            # 参考模型：不计算梯度
            with torch.no_grad():
                ref_logits = ref_model(input_ids=x, attention_mask=(x != tokenizer.pad_token_id).long()).logits  # (2B, T, V)
            ref_log_probs = logits_to_log_probs(ref_logits[:, :-1], y[:, 1:])  # (2B, T-1)

            # 策略模型
            policy_out    = model(input_ids=x, attention_mask=(x != tokenizer.pad_token_id).long())
            policy_logits = policy_out.logits                                   # (2B, T, V)
            policy_log_probs = logits_to_log_probs(policy_logits[:, :-1], y[:, 1:])  # (2B, T-1)

            # DPO Loss（对应 response 部分的 mask 也去掉第一个 token 对齐）
            loss = dpo_loss(ref_log_probs, policy_log_probs, mask[:, 1:], beta=beta)
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        if step % args.log_interval == 0 or step == iters - 1:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_lr   = optimizer.param_groups[0]['lr']
            eta_min = spend_time / (step - start_step) * (iters - step) / 60 if step > start_step else 0
            Logger(
                f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), '
                f'dpo_loss: {current_loss:.4f}, lr: {current_lr:.2e}, '
                f'eta: {eta_min:.1f}min'
            )
            if wandb:
                wandb.log({"dpo_loss": current_loss, "learning_rate": current_lr})

        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            model.eval()
            ckp = os.path.join(args.save_dir, f'{args.save_weight}.pth')
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            # 保存完整训练状态（支持续训）
            torch.save({
                'model':     {k: v.cpu() for k, v in state_dict.items()},
                'optimizer': optimizer.state_dict(),
                'scaler':    scaler.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch':     epoch,
                'step':      step,
            }, os.path.join(args.save_dir, f'{args.save_weight}_ckp.pt'))
            Logger(f'Checkpoint saved → {ckp}')
            model.train()
            del state_dict

        del x_chosen, x_rejected, y_chosen, y_rejected, mask_chosen, mask_rejected
        del x, y, mask, ref_logits, ref_log_probs, policy_out, policy_logits, policy_log_probs, loss


# ─────────────────────────────────────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standard HuggingFace DPO Training")
    # 模型 & 数据
    parser.add_argument("--model_name",      type=str,   default="Qwen/Qwen2.5-7B-Instruct", help="HuggingFace 模型名或本地路径")
    parser.add_argument("--data_path",       type=str,   default="../dataset/dpo.jsonl",      help="DPO JSONL 数据路径")
    # 保存
    parser.add_argument("--save_dir",        type=str,   default="../out",    help="模型保存目录")
    parser.add_argument("--save_weight",     type=str,   default="dpo",       help="保存权重前缀名")
    parser.add_argument("--save_interval",   type=int,   default=100,         help="保存间隔（steps）")
    # 训练超参
    parser.add_argument("--epochs",          type=int,   default=1,           help="训练轮数")
    parser.add_argument("--batch_size",      type=int,   default=2,           help="Batch size")
    parser.add_argument("--learning_rate",   type=float, default=5e-7,        help="学习率（建议 1e-7~5e-6）")
    parser.add_argument("--beta",            type=float, default=0.1,         help="DPO β 温度系数")
    parser.add_argument("--max_seq_len",     type=int,   default=1024,        help="最大序列长度")
    parser.add_argument("--accumulation_steps", type=int, default=1,          help="梯度累积步数")
    parser.add_argument("--grad_clip",       type=float, default=1.0,         help="梯度裁剪阈值")
    parser.add_argument("--log_interval",    type=int,   default=10,          help="日志打印间隔（steps）")
    # 设备 & 精度
    parser.add_argument("--device",          type=str,   default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype",           type=str,   default="bfloat16",  choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--num_workers",     type=int,   default=4,           help="DataLoader 线程数")
    # 续训
    parser.add_argument("--from_resume",     type=str,   default="",          help="续训 checkpoint 路径（.pt 文件），留空则从头训练")
    # 可选加速
    parser.add_argument("--use_compile",     type=int,   default=0, choices=[0, 1], help="是否使用 torch.compile")
    # WandB / SwanLab
    parser.add_argument("--use_wandb",       action="store_true")
    parser.add_argument("--wandb_project",   type=str,   default="DPO-Training")
    args = parser.parse_args()

    # ── 1. 分布式 & 随机种子 ──────────────────────────────────────────────────
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # ── 2. 目录 ───────────────────────────────────────────────────────────────
    os.makedirs(args.save_dir, exist_ok=True)

    # ── 3. 混合精度 ───────────────────────────────────────────────────────────
    device_type  = "cuda" if "cuda" in args.device else "cpu"
    dtype_map    = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype        = dtype_map[args.dtype]
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    scaler       = torch.cuda.amp.GradScaler(enabled=(args.dtype == "float16"))

    # ── 4. WandB ─────────────────────────────────────────────────────────────
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb.init(project=args.wandb_project,
                   name=f"DPO-{args.model_name.split('/')[-1]}-bs{args.batch_size}-lr{args.learning_rate}")

    # ── 5. 模型 & Tokenizer ───────────────────────────────────────────────────
    Logger(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 策略模型（可训练）
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=dtype, trust_remote_code=True
    ).to(args.device)
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger("torch.compile enabled")
    Logger(f'策略模型总参数量：{sum(p.numel() for p in model.parameters()) / 1e6:.3f} M')

    # 参考模型（冻结，不更新梯度）
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=dtype, trust_remote_code=True
    ).to(args.device).eval().requires_grad_(False)
    Logger(f'参考模型总参数量：{sum(p.numel() for p in ref_model.parameters()) / 1e6:.3f} M')

    # ── 6. 数据 & 优化器 ──────────────────────────────────────────────────────
    train_ds = DPODataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    import math as _math
    iters = _math.ceil(len(train_ds) / args.batch_size)
    total_steps = (iters // args.accumulation_steps) * args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.learning_rate / 10)

    # ── 7. 续训恢复 ───────────────────────────────────────────────────────────
    start_epoch, start_step = 0, 0
    if args.from_resume and os.path.isfile(args.from_resume):
        Logger(f"Resuming from checkpoint: {args.from_resume}")
        ckp_data = torch.load(args.from_resume, map_location=args.device)
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        scheduler.load_state_dict(ckp_data['scheduler'])
        start_epoch = ckp_data.get('epoch', 0)
        start_step  = ckp_data.get('step',  0)

    # ── 8. DDP 包装 ───────────────────────────────────────────────────────────
    if dist.is_initialized():
        model     = DistributedDataParallel(model,     device_ids=[local_rank])
        ref_model = DistributedDataParallel(ref_model, device_ids=[local_rank])

    # ── 9. 开始训练 ───────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch)
        indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(
            train_ds, batch_sampler=batch_sampler,
            num_workers=args.num_workers, pin_memory=True
        )
        if skip > 0:
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前 {skip} 个 step，从 step {skip + 1} 开始')
        train_epoch(epoch, loader, len(loader) + skip, ref_model, scheduler,
                    start_step=skip, wandb=wandb, beta=args.beta)

    # ── 10. 清理分布式进程 ────────────────────────────────────────────────────
    if dist.is_initialized():
        dist.destroy_process_group()