# 研究方向纠偏与 Oracle Evaluator 设计

日期：2026-10-17。依据用户最新明确的研究意图修订；这份文件覆盖《二轮差距审查报告》中关于“以序贯来源代理为主线、baseline仅为独立参考”的走向建议。原报告中的可复现代码缺陷和奖励数学分析仍有效。

补充：针对后续“D3：共享noop终端反事实”提案，见 [D3方案复核](C:/project/mythesis/memory_r1_reproduction/D3方案复核.md)。该项在当前GRPO下退化为QA缩放/混合奖励权重调整，不能直接作为独立信用分配贡献。下文的语义支持检查是针对“证据保留”目标的设计路径；若主张限定为固定Answer下的可答性，不强制额外增加语义判别模型，但仍需有效的干预设计与实验验证。

## 1. 必须恢复的研究目标

- `baseline-lora` 是必做基准，也是 evidence 新方法的工程开发基础。它与原始 Memory-R1 差别较大不影响其作为明确命名的实验基准；需要如实声明这些差别。
- evidence 的新增模块是训练期 `Original dialogue + question → Oracle Evaluator → evidence / memory coverage`，不是把轮号覆盖改名后当作该模块已经实现。
- 实际 Answer 仍只读 question + retrieved memory；Manager 的部署输入不增加 QA gold、未来原文或 Oracle 输出。
- 原大纲“不训练新 extractor/retriever、不实现 local rerollout”的范围约束继续有效。
- 无需用户在“原目标”与“缩水的来源代理目标”之间重新二选一。按原目标推进，语义支持评估属于要补的工作。

当前核验：evidence HEAD 为 `332aa055fd194da157ca0c4458ba591af40a65e8`；baseline HEAD 为 `c5d30df6f0301646eaf241c4de385a3b3a4109ea`。evidence 仍是空 bank 序贯重放与来源覆盖；baseline 是单步共享状态训练。当前没有完成上述 Oracle 路径，本文件也不声称已经实现或验证它。

## 2. “对照 Answer Agent”的正确含义

采用同一冻结模型的两种上下文条件，可以不新增一份可训练模型：

```text
原始对话的合法前缀 D≤τ + question
    → 冻结的原文回答/证据提取调用
    → 原文答案 yD + 可定位引文 + 原子事实 E(q)

Manager 产生的 memory + question
    → 原有检索 + 冻结 Answer
    → 记忆答案 yM

E(q) + memory文本/实际检索文本
    → 冻结的支持关系检查
    → 哪些必要事实仍被保留、哪些丢失或被矛盾改写

QA奖励 + 经验证的证据保留辅助奖励
    → 只更新 Manager
```

原文回答支路是用户所说的“对照 Answer Agent”；为了输出 memory coverage，评估过程还必须读取 memory 的内容，不能仅凭 `D+q` 就知道 memory 是否保留了证据。两种上下文采用固定解码与尽量相同的回答要求，记录输入长度与截断。模型可以共享权重、顺序调用；不要求分别训练两个 Answer。

Oracle 是训练期参考评估器，不是绝对真值。须与 QA gold、已有 evidence 标注和人工检查对照。两条支路答对不代表所有证据都保留；两条支路答案一致也不代表正确。

| 控制回答（token-safe raw prefix 或 teacher facts） | memory回答 | 可以诊断什么 |
|---|---|---|
| 正确 | 错误 | 存在记忆路径损失；还需定位是抽取、存储、检索还是读记忆失败 |
| 错误 | 错误 | 不能仅归罪 Manager；先检查控制上下文是否截断、facts 是否完整、模型与评分 |
| 正确 | 正确 | 此题最终可答；不等于证据内容全部保留 |
| 错误 | 正确 | 原文上下文干扰/截断、压缩帮助或模型随机误差等均可能，不能强制惩罚记忆回答 |

## 3. 控制回答的输入契约与奖励公式

“原始对话回答错误”只有在控制输入满足长度与证据完整性后才有诊断意义。代码应优先构造因果 teacher-fact context；若使用原文前缀，必须采用与 Answer 相同的 tokenizer、最大长度、左截断规则，并记录 `control_truncated` 与证据是否仍在上下文中。若发生截断，先通过证据定位压缩或 teacher facts 重建控制输入，不能直接把该 QA 标为无信号。

对同一 QA 定义：

\[
s_T=F1(A_{sub}(q,C_T),y^*),\quad
s_0=F1(A(q,M_0),y^*),\quad
s_i=F1(A(q,R(M_i)),y^*).
\]

其中 `C_T` 是 token-safe 的 causal teacher facts（也可为不截断的原文证据上下文），`M_0` 是窗口起始 bank，`M_i` 是第 i 条 Manager 轨迹的 bank。设 `g_q=1[s_T≥τ_T]`，主训练奖励写为：

\[
r_{i,t}=g_q\left(\eta d_{i,t}-c_{invalid}n_{i,t}\right),\quad t<\tau,
\]

\[
r_{i,\tau}=g_q\left(s_i+\eta d_{i,\tau}-c_{invalid}n_{i,\tau}\right).
\]

`s_T` 的作用是确认控制 Answer 在可用事实下有能力回答，并让控制失败的题跳过 Manager 更新；`s_0` 和 `s_i-s_0` 作为诊断，不直接加到当前 GRPO 的候选奖励。这样不会把 Answer/控制上下文失败错误归罪给 Manager。

控制诊断记录：`s_T` 高而 `s_i` 低表示 Manager memory 路径损失；`s_T` 高且 `s_0` 高表示当前题不需要 Manager 新增信息；`s_T` 高且 `s_i<s_0` 表示 Manager 破坏了可答性；`s_T` 低表示控制上下文或 Answer 不可靠。

该门控不是“sub-answer 分数减法奖励”，而是把控制回答用于判断奖励是否可信；适合当前 GRPO 的组归一化。

## 4. 为什么不能只把两个答案分数相减

固定 q 与原始对话时，原文分数 `RD` 对同一组 Manager 候选是常数。若奖励为 `RM−RD`，组内归一化时常数消失，与使用 RM 得到同样的相对优势（在相同分组、相同其他奖励与相同标准化规则下）。若反向最大化 `RD−RM`，更会鼓励 memory 回答变差。

所以双路评答本身有诊断价值，但“增加对照回答”不自动等于“增加新的候选排序”。若未来确实要优化 Manager 的“超过 noop 的边际收益”，应使用明确的非线性 improvement utility 或按动作进行受控干预，并另做消融；不能把共享常数差值直接称为 COMA 式信用分配。

## 4. 可检验的语义覆盖定义

原文支路先提取少量可定位的原子事实，每项包含原文引文、turn/span位置、事实内容；必须验证引文确实存在于所给前缀，不能直接用 gold answer 编造事实。已有 LoCoMo evidence 可用于训练监督与评估器校验，不用它代替语义支持判断。

支持检查器判定 memory 中的文本是否足以支持每条事实，可输出 `supported / contradicted / insufficient` 并给出 memory id 和引文。含糊结果不强行计作支持。先用明确的 supported 比例作为候选覆盖定义，检查它与人工判断、QA表现的关系，再决定是否进入奖励；不预设分数可信。

分别记录 bank语义覆盖与实际检索上下文语义覆盖，以区分“保存了但检索不到”和“已经丢失”。二者是否混合及权重需要消融，不能自动继承旧的0.5/0.5。

来源ID继续作为追踪和候选定位字段，不能单凭ID交集决定支持。跨表述的时间、实体、否定和UPDATE矛盾是重点测试项。Oracle缓存键至少包含数据前缀、question、模型/prompt版本；候选memory的支持判定还须包含memory内容。保存评估理由供抽查，不让这些内容进入部署策略提示。

## 5. 从 baseline 开发与实验归因

第一步应恢复与 baseline 一致的训练/数据/检索/Answer/验证协议，再通过明确开关增加 Oracle 信号。MiniLM在这里是继承baseline协议的一部分；“不必为了旧序贯分支而强行迁移”的历史判断不等于现在可以无视用户要求的共同基座。

建议主比较：

1. B0：锁定的 baseline-lora 基准。
2. B1：基于同一实现，仅增加原文对照和Oracle日志、不加辅助奖励。用于验证新增路径未意外改变训练；严格比较需隔离Oracle调用对随机数流的影响。
3. B2：同一基础 + 经校验的语义证据辅助奖励。
4. 机制消融：来源ID代理 vs 语义覆盖、bank覆盖 vs 实际检索覆盖等，按先导结果选择必要项。

若单步训练使用 `C(M_after)−C(M_before)`，before项在同状态组中是常数；它不能被包装成已解决长程信用分配。该方法可以研究“证据辅助的记忆动作优化”。如论文必须验证多步per-turn贡献，再增加序贯版本及其不加Oracle奖励的匹配对照，baseline仍保留。不得把新增序贯成本设成Oracle主线的先决条件。

旧R3-T只服务于旧序贯势差方案的机制归因；它不是实现原文Oracle的首要阻塞项。当前优先级应改为：共同baseline基座 → 双路评答与证据支持检查 → 评估器可靠性验证 → 受控训练与消融。

## 6. 文献支持到什么程度

以下是可定位的相关工作，不代表已经完成针对本任务的文献系统综述；本轮未联网核验全文。它们支持组件设计的研究背景，不保证该组合的有效性。

- [FEVER, Thorne et al., 2018](https://arxiv.org/abs/1803.05355)：基于证据判定支持、反驳、信息不足；用于支持关系检查的任务范式。
- [HotpotQA, Yang et al., 2018](https://arxiv.org/abs/1809.09600)：多跳问答的supporting facts监督；说明证据质量可以独立于最终答案评估。
- [Self-RAG, Asai et al., 2023](https://arxiv.org/abs/2310.11511)：区分检索相关性、支持性与回答质量；不是来源ID覆盖的依据，也不是本方案直接复现。
- [Asymmetric Actor Critic, Pinto et al.](https://arxiv.org/abs/1710.06542)：训练评估侧利用部署策略不可见信息的范式参考；与当前LLM评估奖励有区别。
- [DeepSeekMath, Shao et al., 2024](https://arxiv.org/abs/2402.03300)：组相对优势的基础；解释为何同组原文分数常数不会创造新的候选区分。
- [Memory-R1](https://arxiv.org/abs/2508.19828)：原方法参考；baseline-lora与新增Oracle应分别声明实现偏差和方法扩展。

需要的实验依据是：人工核验子集上证据提取与支持判断的准确性、垃圾ADD/错误UPDATE/否定改写等反例表现、原文与memory双路回答的配对结果，以及固定baseline协议下的多种子训练对照。没有这些结果之前，不能称Oracle可信或新方法已有效。

## 7. 本轮处理边界

本轮修正研究设计与旧报告，不直接用未经验证的LLM judge重写奖励，也不切换/覆盖当前分支。此前parser、EOS、日志元数据等工程修复可在共同基座上按适用性保留；全前缀数据与旧势函数模式属于旧序贯实验条件，不再自动代表用户的新方法。
