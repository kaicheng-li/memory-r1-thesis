# baseline-lora：生成参数修复与可选 window24

默认仍为 fullprefix：历史 facts 已在 old_bank 中，Manager 只学习当前一次操作。
F1、GRPO、检索、数据划分、选模和 Manager → Answer 两阶段顺序不变。

`--history-window 24` 只将单步状态的 old_bank 限定为
`max(0, target_turn - 24) <= source_turn < target_turn`。
当前轮 facts 单独输入，不计入这 24 轮；不足 24 轮使用已有历史。
不按 QA 证据删题，也不把跨窗口证据补进 bank。
完整对话 timelines、roll-forward 验证和 Answer 数据构造仍按全对话执行。

## 开启 window24

在项目根目录运行（路径按服务器修改）：

```bash
python3 scripts/rebuild_manager_states.py --history-window 24 --splits train validation
python3 scripts/precheck_counterfactual.py --states data/stage0_window24/manager_states_train.jsonl --model-path /root/autodl-tmp/Qwen2.5-7B-Instruct --output data/stage0_window24/precheck_train_f1.json --device cuda
HISTORY_WINDOW=24 MODEL_PATH=/root/autodl-tmp/Qwen2.5-7B-Instruct bash scripts/run_MemR1.sh
```

训练仍要求预检通过现有 gate；本次不改门槛。
`HISTORY_WINDOW=24` 自动使用 `data/stage0_window24`，并给 Manager/Answer
输出目录及 Answer 数据文件增加 `_window24`，避免默认复用 fullprefix 产物。
自定义路径时，也应分别指定新数据、预检和输出路径。
不设 `HISTORY_WINDOW` 时，原 fullprefix 路径和行为保留。

## 出厂生成参数

Manager/Answer 训练、预检审计及旧模块的 PyTorch 采样入口显式设置：
`top_k=0, top_p=1.0, typical_p=1.0, repetition_penalty=1.0, no_repeat_ngram_size=0`。
部署建库和独立评答的 greedy 入口显式关闭两种复读限制。
默认训练 temperature 仍为 1；greedy 仍为 `do_sample=False`。
未改变 KV cache、梯度检查点、token 上限、prompt、奖励或优化器。
某些 Transformers 版本仍会提示 greedy 忽略采样参数；不为消除此提示再次重构生成逻辑。

旧预检若使用了出厂惩罚或截断，需用修复后的生成入口重新产生预测；仅重新算 F1 无法修复旧预测。
window24 也必须使用对应状态的预检，不能沿用 fullprefix 预检。
旧权重和训练产物保留，但不是修复后条件下的新实验结果。
