import fs from 'node:fs';
import path from 'node:path';

const root = path.resolve('E:/infoseek/实验归档/18_增强攻击实验v8_ASTRA');
const astraRoot = path.join(root, 'results_max500_20261010');
const b3Root = path.resolve('E:/infoseek/实验归档/17_攻击强度扫描实验设计/wb_gale_budget_b3_500');
const tSnrRoot = path.join(root, 'trajectory_snr_500_20261010');
const sets = ['nq', 'hotpotqa'];

function readJsonl(file) {
  return fs.readFileSync(file, 'utf8').trim().split(/\r?\n/).filter(Boolean).map(JSON.parse);
}
function allRecords(dir) {
  return sets.flatMap(dataset => readJsonl(path.join(dir, dataset, 'records.jsonl')));
}
function mean(xs) { return xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null; }
function median(xs) {
  if (!xs.length) return null;
  const ys = [...xs].sort((a, b) => a - b), m = Math.floor(ys.length / 2);
  return ys.length % 2 ? ys[m] : (ys[m - 1] + ys[m]) / 2;
}
function pct(x) { return x == null ? '—' : `${(100 * x).toFixed(2)}%`; }
function fixed(x, n = 2) { return x == null ? '—' : Number(x).toFixed(n); }
function wilson(k, n, z = 1.96) {
  const p = k / n, z2 = z * z, den = 1 + z2 / n;
  const c = (p + z2 / (2 * n)) / den;
  const d = z * Math.sqrt((p * (1 - p) + z2 / (4 * n)) / n) / den;
  return [c - d, c + d];
}
function ciPairedDiff(a, b, rounds = 20000) {
  // Deterministic bootstrap of paired Bernoulli differences, in percentage points.
  let state = 20261010 >>> 0;
  const rand = () => { state = (state + 0x6D2B79F5) >>> 0; let t = state; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  const n = a.length, vals = new Array(rounds);
  for (let r = 0; r < rounds; r++) {
    let d = 0;
    for (let i = 0; i < n; i++) { const j = Math.floor(rand() * n); d += (a[j] ? 1 : 0) - (b[j] ? 1 : 0); }
    vals[r] = d / n;
  }
  vals.sort((x, y) => x - y);
  return [vals[Math.floor(.025 * rounds)], vals[Math.floor(.975 * rounds)]];
}
function snrDb(sim) {
  if (!Number.isFinite(sim)) return null;
  const noise = 2 * (1 - Math.min(1, sim));
  return noise <= 1e-12 ? Infinity : -10 * Math.log10(noise);
}
function groupSummary(rows) {
  const success = rows.filter(x => x.success);
  const get = (name, subset = rows) => subset.map(x => x.best?.[name]).filter(Number.isFinite);
  const sims = get('embedding_similarity');
  const snrs = sims.map(snrDb).filter(Number.isFinite);
  const ops = {};
  for (const r of success) for (const op of (r.best?.operators || [])) ops[op] = (ops[op] || 0) + 1;
  return {
    n: rows.length, success: success.length, asr: success.length / rows.length, ci95: wilson(success.length, rows.length),
    mean_retrieval: mean(rows.map(x => x.retrieval_calls)), mean_target: mean(rows.map(x => x.target_calls)),
    mean_edit: mean(get('edit_ratio')), median_edit: median(get('edit_ratio')),
    median_similarity: median(sims), mean_similarity: mean(sims),
    median_snr_db: median(snrs), mean_snr_db: mean(snrs),
    median_drift: median(get('retrieval_drift')), mean_drift: mean(get('retrieval_drift')),
    operators: Object.entries(ops).sort((a, b) => b[1] - a[1])
  };
}

const astra = allRecords(astraRoot), b3 = allRecords(b3Root);
const b3ById = new Map(b3.map(r => [`${r.dataset}:${r.example_id}`, r]));
if (astra.length !== 500 || b3.length !== 500) throw new Error(`Expected 500 paired records; got ASTRA=${astra.length}, B3=${b3.length}`);
const paired = astra.map(a => ({ a, b: b3ById.get(`${a.dataset}:${a.example_id}`) }));
if (paired.some(x => !x.b)) throw new Error('Sample IDs are not fully paired.');
const both = paired.filter(x => x.a.success && x.b.success).length;
const astraOnly = paired.filter(x => x.a.success && !x.b.success).length;
const b3Only = paired.filter(x => !x.a.success && x.b.success).length;
const neither = paired.length - both - astraOnly - b3Only;
const astraS = groupSummary(astra), b3S = groupSummary(b3);
const byDataset = Object.fromEntries(sets.map(ds => [ds, { astra: groupSummary(astra.filter(x => x.dataset === ds)), b3: groupSummary(b3.filter(x => x.dataset === ds)) }]));
const pairedCI = ciPairedDiff(paired.map(x => x.a.success), paired.map(x => x.b.success));
const tSnrPath = path.join(tSnrRoot, 'trajectory_snr_summary.json');
const tSnr = fs.existsSync(tSnrPath) ? JSON.parse(fs.readFileSync(tSnrPath, 'utf8')) : null;
const report = {
  generated_at: new Date().toISOString(),
  sample: { total: 500, nq: 166, hotpotqa: 334, seed: 20261002, criterion: 'clean-correct; strict attacked EM/F1 both zero' },
  astra: astraS, b3: b3S, by_dataset: byDataset,
  paired: { both, astra_only: astraOnly, b3_only: b3Only, neither, delta_asr: astraS.asr - b3S.asr, delta_ci95_bootstrap: pairedCI },
  target_model_trajectory_snr: tSnr,
  environment: JSON.parse(fs.readFileSync(path.join(astraRoot, 'nq', 'environment.json'), 'utf8'))
};
fs.writeFileSync(path.join(root, 'analysis_max500.json'), JSON.stringify(report, null, 2));

const bars = (a, b) => `<div class="bars"><span class="b old" style="width:${100*a}%"></span><span class="b new" style="width:${100*b}%"></span></div>`;
const row = (name, oldS, newS) => `<tr><td>${name}</td><td>${oldS.success}/${oldS.n} (${pct(oldS.asr)})</td><td>${newS.success}/${newS.n} (${pct(newS.asr)})</td><td>${((newS.asr-oldS.asr)*100).toFixed(2)} pp</td><td>${bars(oldS.asr,newS.asr)}</td></tr>`;
const opRows = astraS.operators.slice(0, 10).map(([o,n]) => `<tr><td><code>${o}</code></td><td>${n}</td><td>${pct(n/astraS.success)}</td></tr>`).join('');
const html = `<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ASTRA v8 · 500题正式评测</title>
<style>
body{margin:0;background:#f5f7fb;color:#18212f;font:15px/1.65 system-ui,-apple-system,"Segoe UI",Arial,sans-serif}.wrap{max-width:1080px;margin:auto;padding:42px 22px 70px}h1{font-size:30px;line-height:1.2;margin:0 0 8px}h2{font-size:20px;margin:34px 0 12px}.sub{color:#627087;margin:0}.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:24px 0}.card,.panel{background:white;border:1px solid #e2e8f0;border-radius:12px;padding:18px;box-shadow:0 1px 2px #0f172a08}.k{font-size:12px;color:#64748b}.v{font-size:25px;font-weight:750;margin-top:3px}.accent{color:#0969da}table{border-collapse:collapse;width:100%;background:white;border:1px solid #e2e8f0;border-radius:12px;overflow:hidden}td,th{padding:11px 12px;border-bottom:1px solid #e9eef5;text-align:left;vertical-align:middle}th{font-size:12px;color:#536277;background:#f8fafc}.bars{height:11px;width:120px;display:flex;gap:2px;align-items:center}.b{height:100%;border-radius:4px;min-width:2px}.old{background:#b8c3d6}.new{background:#1677ff}code{background:#f1f5f9;border-radius:4px;padding:2px 4px;font-size:.9em}.note{border-left:4px solid #f59e0b;background:#fffbeb;padding:13px 15px;margin-top:15px;border-radius:0 8px 8px 0}.ok{color:#087f5b;font-weight:650}.small{font-size:13px;color:#58667a}@media(max-width:700px){.grid{grid-template-columns:repeat(2,1fr)}.wrap{padding:26px 13px}}
</style><main class="wrap">
<p class="sub">实验归档 / 18_增强攻击实验v8_ASTRA · 生成于 ${new Date().toLocaleString('zh-CN',{timeZone:'Asia/Shanghai'})}</p><h1>ASTRA：动作轨迹敏感的白盒离散攻击</h1><p class="sub">500 个既有 clean-correct 样本的端到端正式评测。目标生成器、检索器、语料、向量索引与评测规则均保持不变。</p>
<section class="grid"><div class="card"><div class="k">严格 ASR</div><div class="v accent">${pct(astraS.asr)}</div><div class="small">${astraS.success} / ${astraS.n}，Wilson 95% CI ${pct(astraS.ci95[0])}–${pct(astraS.ci95[1])}</div></div><div class="card"><div class="k">相对 B3 的绝对提升</div><div class="v accent">+${((astraS.asr-b3S.asr)*100).toFixed(2)} pp</div><div class="small">配对 bootstrap 95% CI +${(pairedCI[0]*100).toFixed(2)}–+${(pairedCI[1]*100).toFixed(2)} pp</div></div><div class="card"><div class="k">平均检索 / 目标调用</div><div class="v">${fixed(astraS.mean_retrieval,1)} / ${fixed(astraS.mean_target,1)}</div><div class="small">上限 144 / 12；另有一次仅测量用 clean 动作探针</div></div><div class="card"><div class="k">成功样本中的算子命中</div><div class="v">${astraS.operators[0]?.[0] || '—'}</div><div class="small">出现于 ${astraS.operators[0]?.[1] || 0} 个成功样本</div></div></section>
<h2>端到端攻击效果</h2><table><thead><tr><th>数据集</th><th>v7 B3</th><th>v8 ASTRA</th><th>差值</th><th>比例</th></tr></thead><tbody>${row('合计',b3S,astraS)}${row('NQ',byDataset.nq.b3,byDataset.nq.astra)}${row('HotpotQA',byDataset.hotpotqa.b3,byDataset.hotpotqa.astra)}</tbody></table>
<h2>配对结果</h2><div class="panel"><p><b>共同成功</b> ${both}　<b>ASTRA 新增成功</b> <span class="ok">${astraOnly}</span>　<b>B3 独有成功</b> ${b3Only}　<b>共同失败</b> ${neither}</p><p class="small">两次运行使用同一批 500 个题目与相同种子。配对提升不代表仅由某一个模块导致：ASTRA 同时改变了候选生成、白盒目标、搜索轮数/beam 和预算，并取消了 SNR 下限。</p></div>
<h2>攻击强度与成本</h2><table><thead><tr><th>度量</th><th>ASTRA v8</th><th>解释</th></tr></thead><tbody><tr><td>平均 / 中位字符编辑率</td><td>${pct(astraS.mean_edit)} / ${pct(astraS.median_edit)}</td><td>相对原问题的 Levenshtein 字符编辑比例</td></tr><tr><td>中位 e5 余弦相似度</td><td>${fixed(astraS.median_similarity,4)}</td><td>仅作语义代理；本 profile 不以其过滤</td></tr><tr><td>中位语义 SNR</td><td>${fixed(astraS.median_snr_db,2)} dB</td><td>由 e5 余弦相似度推得；不是端到端感知 SNR</td></tr><tr><td>平均 / 中位检索漂移</td><td>${fixed(astraS.mean_drift,3)} / ${fixed(astraS.median_drift,3)}</td><td>Top-5 检索集合的加权变化代理</td></tr><tr><td>平均目标调用</td><td>${fixed(astraS.mean_target,2)}</td><td>正式候选评估；不含一次 clean 动作测量</td></tr></tbody></table>
${tSnr ? `<h2>目标模型端到端轨迹 SNR（T-SNR）</h2><table><thead><tr><th>对象</th><th>有限值均值</th><th>中位数</th><th>中位轨迹余弦</th><th>∞ 数量</th></tr></thead><tbody><tr><td>全部 500 条</td><td>${fixed(tSnr.all.mean_t_snr_db,2)} dB</td><td>${fixed(tSnr.all.median_t_snr_db,2)} dB</td><td>${fixed(tSnr.all.median_trajectory_cosine,4)}</td><td>${tSnr.all.infinite_t_snr_count}</td></tr><tr><td>189 条严格成功攻击</td><td>${fixed(tSnr.successful_attacks.mean_t_snr_db,2)} dB</td><td>${fixed(tSnr.successful_attacks.median_t_snr_db,2)} dB</td><td>${fixed(tSnr.successful_attacks.median_trajectory_cosine,4)}</td><td>${tSnr.successful_attacks.infinite_t_snr_count}</td></tr></tbody></table><div class="note"><b>T-SNR 定义：</b>将实际观测到的「问题 → 结构化工具动作 → 每步 Top-3 文档 ID/标题/分数 → 最终答案」送入<b>冻结的目标生成器</b>，对末层 token 表示均值池化并 L2 归一化为 (z)，计算 −10 log10(‖z_clean−z_attack‖²)。它是目标模型空间中的端到端行为信号强度，排除了思维链和金答案；不是声学 SNR，也不同于 e5 输入语义 SNR。∞ 只表示该序列化轨迹的数值表示未产生可分辨差异，不能推出系统的所有内部状态完全相同。</div>` : ''}
<div class="note"><b>可比性边界：</b>本次为 <code>max-asr</code> profile，明确关闭 e5-SNR 下限并放宽编辑/重叠门控。因此 37.80% 描述的是该设定下的最大攻击能力，不能作为“18 dB 隐蔽攻击”的结果。若要建立严格因果结论，应固定预算和门控，分别消融动作 NLL、梯度定位与候选族。</div>
<h2>成功样本中的候选算子（可重叠）</h2><table><thead><tr><th>算子</th><th>成功样本出现次数</th><th>占 189 个成功样本</th></tr></thead><tbody>${opRows}</tbody></table>
<h2>方法与审计</h2><div class="panel"><p><b>ASTRA</b> 用冻结目标生成器的白盒梯度定位词级脆弱位置；同时优化干净答案的 NLL、首个工具动作轨迹的 NLL、检索边界代理和多样性。候选均为离散文本变换，包括梯度定位的词内边界、原有锚点回显、疑问词微改写、引号载体等。</p><p><b>成功定义：</b>样本首先必须为 clean-correct；攻击后输出相对金答案的 EM 与 F1 均为 0 才计作成功。未修改模型权重、语料或索引；不向候选或提示中注入金答案。</p><p class="small">原始逐题数据：<code>results_max500_20261010/{nq,hotpotqa}/records.jsonl</code>；T-SNR 原始数据：<code>trajectory_snr_500_20261010/trajectory_snr.jsonl</code>。可重算汇总：<code>analysis_max500.json</code>；生成脚本：<code>build_report.mjs</code>。</p></div>
</main></html>`;
fs.writeFileSync(path.join(root, '实验报告_ASTRA_v8_500.html'), html);
console.log(JSON.stringify(report, null, 2));

