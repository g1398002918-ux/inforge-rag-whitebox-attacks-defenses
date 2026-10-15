# v8 ASTRA 实验归档

本目录独立于 v5、v6 和 v7，包含 ASTRA 白盒离散攻击的实现、500 题正式结果与可复算报告。

- `METHOD.md`：攻击方法、威胁模型和审计要求。
- `src/astra.py`：攻击搜索实现；`server_astra_runner.py`：服务器运行器。
- `results_max500_20261010/`：500 条逐题原始记录（NQ 166、HotpotQA 334）。
- `analysis_max500.json`：由原始 JSONL 重算的汇总、配对比较和置信区间。
- `trajectory_snr_500_20261010/`：500 条目标模型端到端轨迹 SNR（T-SNR）原始数据与摘要。
- `server_e2e_trajectory_snr.py`：冻结目标模型的 T-SNR 只读审计脚本。
- `实验报告_ASTRA_v8_500.html`：可直接打开的实验报告。
- `实验报告_ASTRA_v8_全量4056.html`：4,056 题终版全量实验报告。
- `analysis_full4056.json`：全量结果汇总与不可变性审计。
- `build_report.mjs`：报告再生成脚本；仅读取 v8 记录与 v7 B3 对照记录。

正式结果：189 / 500，严格 ASR 37.80%。该数字对应 `max-asr` profile：没有 SNR 下限，故不能与 18 dB 隐蔽攻击结果混为一谈。

端到端 T-SNR：全部 500 条的中位值为 18.23 dB；189 条严格成功攻击的中位值为 16.51 dB（有限值均值 18.00 dB）。它衡量目标模型对完整可观测轨迹的表示变化，不等同于输入 e5-SNR。

全量结果：1,589 / 4,056，严格 ASR 39.1765%；NQ 451 / 1,343，HotpotQA 1,138 / 2,713。

