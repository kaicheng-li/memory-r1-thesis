# 数据目录身份说明(权威版)

> 更新:2026-10-17。每个文件的身份、消费方、能否删除。
> 判定依据:run_MemR1.sh / train_manager_grpo.py / precheck_sequential.py / build_sequential_from_stage0.py 的实际读取路径。
> 三份 review 数据的 manifest SHA256 已于 2026-10-17 校验:输出哈希与 stage0 源哈希全部一致。

## ✅ 正式训练/评测数据(当前流水线直接消费,不可删)

| 文件 | 身份 | 消费方 |
|---|---|---|
| `review/manager_tuples_train_fullprefix.jsonl` (+.manifest) | **Manager 正式训练集**(113 组 / 150 QA / 全前缀) | run_MemR1.sh 的 `DATA_PATH` 默认值 |
| `stage0/dialogue_timelines_validation.json` | **验证选模数据**(conv-30 全对话 timeline,81 QA) | `VALIDATION_TIMELINES`,每 epoch roll-forward |
| `stage0/dialogue_timelines_test.json` | **最终 test 评测数据**(8 对话 1307 QA) | evaluation/evaluate_test_final.py |
| `review/manager_tuples_validation_fullprefix.jsonl` (+.manifest) | 验证侧单步/预检诊断数据(66 组 / 81 QA) | 诊断用,保留 |

## 🔁 源头缓存(再生上面文件的唯一原料,不可删)

| 文件 | 身份 |
|---|---|
| `stage0/manager_states_train.jsonl` / `manager_states_validation.jsonl` | QA 绑定与 old_bank 状态(build_sequential_from_stage0 的 `--states` 输入) |
| `stage0/dialogue_timelines_train.json` | train timeline(`--timelines` 输入;正式训练集由它生成) |
| `stage0/stage0_stats.json` | stage0 重建验收数字 |
| `manager_training_data_vllm.jsonl` (182M) | **stage0 的上游**:teacher(vLLM)抽取的全部 10 对话 facts 缓存。当前流水线不直接读,但**重建 stage0 需要它,teacher 不可免费重跑** → 保留(可移入 archive/ 或压缩) |

## 🗑️ archive/(全部可随时整目录删除)

| 文件 | 原因 |
|---|---|
| `manager_tuples_train_window24.jsonl` (+.manifest) | 24 轮短窗口消融/冒烟数据(96 组/122 QA,缺 28 道长程题)。**可一条命令免 GPU 再生**:`python3 scripts/build_sequential_from_stage0.py --timelines data/stage0/dialogue_timelines_train.json --states data/stage0/manager_states_train.jsonl --split train --history-window 24 --output ...`(2026-10-18 移入) |
| `answer_training_data_bootstrap_Qwen2.5-7B-Instruct.jsonl` | 旧 bootstrap 协议产物,过不了现行 SHA 校验 |
| `answer_training_data_qwen25_7b_qlora.jsonl.bak` | 备份残留 |

## 🔀 baseline-lora 分支消费的数据(同一 stage0,不同形态)

| 文件 | 用途 |
|---|---|
| `stage0/manager_states_train.jsonl` | baseline **训练集**(150 QA 单步状态,trainer 直接消费) |
| `stage0/manager_states_validation.jsonl` | baseline 验证第一层(81 QA/66 states,trainer 硬校验) |
| `stage0/dialogue_timelines_validation.json` | baseline 验证第二层 roll-forward(369 轮/81 QA)——与 evidence 分支共用 |
| `stage0/precheck_train_f1.json` | baseline 预检门(**GPU 机已生成并过线,2026-10-16**;与 evidence 的 precheck_sequential 输出**不通用**)。同目录旧的 `precheck_train.json`(F1 改版前)已过时可归档 |
| `locomo10.json` | 最上游原始数据,仅重建 stage0 时需要 |
| `answer_training_Qwen2.5-7B-Instruct_{train,validation}.jsonl`(+.manager_trace.jsonl,GPU 机) | baseline Algorithm-2 产物(trained Manager 重放生成),2026-10-17 生成 |

同源关系:locomo10 + vllm facts → stage0 → baseline 直接吃 manager_states_*(单步);evidence 经 build_sequential_from_stage0 再加工成 review/manager_tuples_*(序贯)。R2 与 R1/R3 对照的"数据同源"声明以 stage0 哈希为据。

## 📐 识别规则(以后不用猜)

1. **有 .manifest.json 且 SHA 可验 = 正式产物**;manifest 里的 `history_window: -1` = 全前缀正式版,`24` = 消融版。
2. **训练会自动留痕**:每次 run 的 `run_manifest.json` 记录消费的数据 SHA256——事后想知道某次训练用了哪份数据,看 manifest,不看文件名。
3. **Answer 数据合法性由脚本判定**:run_MemR1.sh 校验 `metadata.manager_tuples_sha256` 与当前 DATA_PATH 哈希一致,不一致自动重建,人工不用管。
4. **预检/冒烟产物**(GPU 机上的 `precheck_*` / smoke 输出目录):只保留**用于校准 CONTROL_THRESHOLD 的那一份**(含 paired.jsonl + summary.json 的完整 run,写进论文);所有 `--limit-tuples` 的冒烟输出可删。precheck 脚本拒绝写入已有 run_manifest.json 的目录,所以每次预检必然是独立新目录,按目录整体保留/删除即可。
