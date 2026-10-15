# ASTRA：动作轨迹感知的白盒离散攻击

## 目标与硬边界

ASTRA（Action-Trace Sensitive Retrieval Attack）追求更高的端到端严格 ASR，但不改变 InForage 生成器权重、e5 权重、向量索引、语料、提示模板或评价程序。金答案只能用于评价，禁止写入候选问题。最终输入始终为离散文本，不使用 PGD、APGD、C&W 或连续 embedding 写入。

本方法分为两个明确标注的配置：

- **Fidelity profile**：保留 v6 的实体/数字、token overlap、编辑率与 SNR 约束，用于与 v6/v7 可比的隐蔽攻击报告。
- **Max-ASR profile**：仍保留实体/数字与禁止答案注入，但取消 SNR 门槛，放宽字符编辑率、词重合与长度增长；它是高扰动攻击，不能称为 18 dB 隐蔽攻击。

任何报告必须分别展示两个 profile，不能用 Max-ASR 的结果替代 Fidelity 的结论。

## 新的白盒目标

v6/WB-GALE 只压低干净最终答案 `y0` 的教师强制概率。ASTRA 还利用干净端到端轨迹第一轮工具动作 `a0`（通常是模型的搜索请求）。对候选问题 `q'`：

```
L_ans(q') = -mean log p_theta(y0 | P(q'))
L_act(q') = -mean log p_theta(a0 | P(q'))
S_trace(q') = ΔL_ans(q') + lambda_act * ΔL_act(q')
```

`a0` 来自干净调用的真实生成轨迹，而不是人为指定的恶意指令。增大 `S_trace` 的含义是同时降低模型复现干净回答与干净第一步检索动作的倾向。每个候选仍要经过真实端到端调用，以 EM=0 且 F1<0.5 判定成功。

## 候选生成

候选只使用原问题已有内容，按白盒显著 token 和原有实体/数字锚点生成：

1. **多位置梯度边界**：在显著长词的 1/3、中点、2/3 处分别测试 Unicode 边界，避免 v6 每词单一中点造成的覆盖饱和。
2. **显著锚点重权**：回显已有高显著关键词/实体锚点，但不引入新事实或答案短语。
3. **分词/格式族**：保守的引号、冒号、括号、空格、Unicode 归一化与问题载体变体；每个操作均记录。
4. **局部等义族**：保留原有 WH 微改写，并只对规则可验证的结构作本地替换。
5. **组合链**：在第一轮高分且互异的父候选上进行第二轮组合；Max-ASR profile 可允许第三轮，但须分别报告深度分布。

## 多目标、多样性选择

候选的基础效用为：

```
U(q') = S_trace(q')
        + lambda_ret * boundary_gain(q')
        + lambda_evd * evidence_drop(q')
        - lambda_edit * edit_ratio(q')
```

从候选池选择送入真实目标的候选时使用 MMR：

```
MMR(q') = U(q') - lambda_dup * max_{z in selected} cosine(e(q'), e(z))
```

并额外惩罚 Top-k 文档高度重合的候选。其目的不是制造更大文本差异，而是避免有限的目标调用预算被一组几乎等价的高分候选消耗。

## 搜索预算与实验分解

建议的 v8 Max-ASR 配置是：3 轮、beam=24、retrieval budget=144、target budget=12、target batch=6。它不应与单因素 SNR 实验混为一谈。

实施顺序：

1. 2 条冒烟：验证动作轨迹提取、离散候选审计、权重哈希不变。
2. 相同 500 条配对样本：Fidelity profile 与 Max-ASR profile 分别运行。
3. 对 Max-ASR 与 B3 做逐样本配对、bootstrap 区间、调用成本和语义指标比较。
4. 只有在 500 条结果存在稳定提升时，才决定是否全量运行。

## 必须记录的审计字段

- 权重/索引只读状态与文件哈希；
- 干净回答、干净第一轮动作、`ΔL_ans`、`ΔL_act`、最终 `U` 与 MMR 选择顺序；
- 每个候选的操作链、编辑率、token overlap、实体/数字检查、SNR、Top-k 文档；
- 目标调用数、检索调用数、轨迹、EM/F1 和失败状态；
- profile 名称。禁止将 Max-ASR 结果标记为高隐蔽性。

