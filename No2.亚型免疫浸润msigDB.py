#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
immune_pipeline_msigdb_by_cluster.py  (bigger fonts, hard-applied Times New Roman)
"""

import os, glob, re, warnings
warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sns
from matplotlib.font_manager import FontProperties, findfont
from gseapy.parser import read_gmt
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from statsmodels.stats.multitest import multipletests

# ========= 参数 =========
rna_pattern   = "cluster_*_RNA.csv"
clin_file     = "clinical.tsv"
surv_file     = "survival.tsv"
gmt_file      = "c7.all.v2025.1.Hs.symbols.gmt"
out_dir       = "No2.immune_msigdb_by_cluster_shorttitle"
score_method  = "mean"        # 'mean' 或 'ssgsea'
min_geneset_n = 5
min_cox_n     = 10
min_km_n      = 20
min_group_n   = 8
top_k_km      = 3
os.makedirs(out_dir, exist_ok=True)

# ========= 字体工具（强制 Times New Roman + 粗体 + 大号）=========
def _get_tnr_bold():
    candidates = ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"]
    for fam in candidates:
        try:
            findfont(FontProperties(family=fam), fallback_to_default=False)
            return FontProperties(family=fam, weight="bold")
        except Exception:
            continue
    return FontProperties(family="serif", weight="bold")

FP_TNR = _get_tnr_bold()
FS_TITLE = 21
FS_LABEL = 21
FS_TICK  = 21
FS_LEG   = 21
FS_ANNO  = 21

# 基础风格（白底+网格）
sns.set_theme(style="whitegrid")

def apply_axes_fonts(ax):
    """把 TNR 粗体和大字号硬套到一个 Axes 上（不受 rc/主题覆盖）"""
    # 标题
    ax.set_title(ax.get_title(), fontproperties=FP_TNR, fontsize=FS_TITLE)
    # 坐标轴标题
    ax.xaxis.label.set_fontproperties(FP_TNR); ax.xaxis.label.set_fontsize(FS_LABEL)
    ax.yaxis.label.set_fontproperties(FP_TNR); ax.yaxis.label.set_fontsize(FS_LABEL)
    # 刻度
    for lab in ax.get_xticklabels():
        lab.set_fontproperties(FP_TNR); lab.set_fontsize(FS_TICK)
    for lab in ax.get_yticklabels():
        lab.set_fontproperties(FP_TNR); lab.set_fontsize(FS_TICK)

def style_legend(ax):
    leg = ax.legend(loc="lower left", frameon=True, framealpha=0.9)
    if leg is not None:
        for txt in leg.get_texts():
            txt.set_fontproperties(FP_TNR); txt.set_fontsize(FS_LEG)
    return leg

# ========= 标题清理 =========
_prefix_pat = re.compile(r'^(IMMUNE|MSIGDB|BIOCARTA|REACTOME|KEGG|GO|HALLMARK|MODULE)\b[_\- ]*', re.I)
_suffix_pat = re.compile(r'(_UP|_DOWN|_DN|_POSITIVE|_NEGATIVE)$', re.I)
_id_pat_any = re.compile(r'(GSE\d+|HSA\d+|\bhsa\d+\b|\bmmu\d+\b|\bRNO\d+\b)', re.I)
_nonword    = re.compile(r'[_\-]+')

def clean_title(name: str) -> str:
    s = name.strip()
    s = _id_pat_any.sub('', s)
    s = _prefix_pat.sub('', s)
    s = _suffix_pat.sub('', s)
    s = _nonword.sub(' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def shorten_title(name: str, max_words=12):
    s = clean_title(name)
    parts = s.split()
    return (' '.join(parts[:max_words]) if parts else name).title()

# ========= 1) 合并表达 & 簇 =========
expr_dfs, cluster_map = [], {}
for fn in sorted(glob.glob(rna_pattern)):
    cid = int(os.path.basename(fn).split("_")[1])
    df  = pd.read_csv(fn, index_col=0).apply(pd.to_numeric, errors="coerce")
    expr_dfs.append(df)
    for sample in df.columns:
        cluster_map[sample] = cid
expr = pd.concat(expr_dfs, axis=1)
print("1) Expression matrix:", expr.shape)

# ========= 2) 读 GMT & 过滤 =========
if not os.path.exists(gmt_file):
    raise FileNotFoundError(f"找不到 {gmt_file}")
gene_sets_all = read_gmt(gmt_file)
gene_sets = {n: list({g for g in gs if g in expr.index}) for n, gs in gene_sets_all.items()}
gene_sets = {n: gs for n, gs in gene_sets.items() if len(gs) >= min_geneset_n}
print(f"2) Loaded {len(gene_sets)} gene sets (>= {min_geneset_n} genes)")

# ========= 3) 基因集分数 =========
if score_method.lower() == "mean":
    scores = pd.DataFrame(index=expr.columns, dtype=float)
    for gs_name, genes in gene_sets.items():
        scores[gs_name] = expr.loc[genes].mean(axis=0)
elif score_method.lower() == "ssgsea":
    import gseapy as gp
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as td:
        expr_t = expr.copy(); expr_t.index.name = 'gene'
        expr_t.to_csv(os.path.join(td, "expr.tsv"), sep="\t")
        gmt_tmp = os.path.join(td, "sets.gmt")
        with open(gmt_tmp, "w") as f:
            for n, genes in gene_sets.items():
                f.write(n + "\tNA\t" + "\t".join(genes) + "\n")
        enr = gp.ssgsea(data=expr_t, gene_sets=gmt_tmp, outdir=td,
                        sample_norm_method=None, no_plot=True, verbose=False)
        scores = enr.res2d.T
        scores.index.name = None
        scores = scores.loc[expr.columns]
else:
    raise ValueError("score_method must be 'mean' or 'ssgsea'")
print("3) Scores matrix:", scores.shape)
scores.to_csv(os.path.join(out_dir, f"msigdb_C7_scores_{score_method}.csv"))

# ========= 4) 合并临床 & 生存 =========
clin = pd.read_csv(clin_file, sep="\t", index_col=0)
surv = pd.read_csv(surv_file,  sep="\t", index_col=0)
def _col(df, cands):
    for c in cands:
        if c in df.columns: return c
    raise KeyError(f"未找到列：{cands}")
time_col  = _col(surv, ["OS.time","OS_time","time","Time"])
event_col = _col(surv, ["OS","event","Event","status","Status"])
pheno = clin.join(surv, how="inner")
pheno["Cluster"] = pheno.index.map(cluster_map)
print("4) Pheno + cluster:", pheno.shape)
pheno.to_csv(os.path.join(out_dir, "pheno_with_clusters.csv"))

# ========= 5) 每簇：z-score → Cox/KM =========
RED, BLUE = "#d62728", "#1f77b4"
km_summaries = []

for cl in sorted([c for c in pd.unique(list(cluster_map.values())) if c in pheno["Cluster"].dropna().unique()]):
    print(f"\n=== Cluster {cl} ===")
    base = pheno[pheno["Cluster"] == cl].dropna(subset=[time_col, event_col])
    idx  = base.index
    if base.shape[0] < min_cox_n:
        print(f" Cluster {cl}: n={base.shape[0]} < {min_cox_n}, skip")
        continue

    # —— 该簇内 z-score —— #
    S = scores.loc[idx]
    mu = S.mean(axis=0)
    sd = S.std(axis=0, ddof=0)
    Z = (S - mu) / sd.replace(0, np.nan)
    Z = Z.dropna(axis=1, how="all")
    Z.to_csv(os.path.join(out_dir, f"zscore_scores_cluster{cl}.csv"))
    data_cl = base.join(Z, how="inner")

    # 5.1 Cox（用 Z）
    rec = []
    for gs in Z.columns:
        df_ct = data_cl[[gs, time_col, event_col]].dropna()
        if df_ct.shape[0] < min_cox_n or df_ct[gs].std(ddof=0) == 0:
            continue
        try:
            cph = CoxPHFitter()
            cph.fit(df_ct, duration_col=time_col, event_col=event_col, show_progress=False)
            hr = float(cph.hazard_ratios_[gs])
            ci_l, ci_u = map(float, cph.confidence_intervals_.loc[gs])
            p = float(cph.summary.loc[gs, "p"])
            rec.append((gs, hr, ci_l, ci_u, p))
        except Exception:
            continue
    cox_df = pd.DataFrame(rec, columns=["gene_set","HR","low95","high95","p"])
    if cox_df.empty:
        print(" No valid Cox results."); continue
    cox_df["adj_p"] = multipletests(cox_df["p"], method="fdr_bh")[1]
    cox_df.to_csv(os.path.join(out_dir, f"cox_cluster{cl}.csv"), index=False)

    # 5.2 森林图（FDR<0.1）
    sig = cox_df[cox_df["adj_p"] < 0.1].sort_values("HR")
    if not sig.empty:
        lo = np.abs(sig["HR"] - sig["low95"]); hi = np.abs(sig["high95"] - sig["HR"])
        plt.figure(figsize=(10.5, max(5.5, len(sig)*0.7))); ax = plt.gca()
        ax.errorbar(x=sig["HR"], y=np.arange(len(sig)), xerr=[lo, hi],
                    fmt='o', color='black', ecolor='gray', capsize=27)
        labels = [shorten_title(gs, 12) for gs in sig["gene_set"]]
        ax.set_yticks(np.arange(len(sig))); ax.set_yticklabels(labels)
        apply_axes_fonts(ax)
        for lab in ax.get_yticklabels():
            lab.set_fontproperties(FP_TNR); lab.set_fontsize(max(FS_TICK, 32))
        ax.axvline(1, linestyle="--", color="red")
        ax.set_xscale("log")
        ax.set_xlabel("Hazard Ratio (per +1 SD)")
        ax.set_title(f"Cluster {cl} Cox Forest (FDR<0.1)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"forest_cluster{cl}.tiff"), dpi=600)
        plt.close()
    else:
        print(" No significant sets for forest plot")

    # 5.3 KM（取 FDR 最小的前 K 个；用 Z；按 HR 定义红/蓝）
    topk = cox_df.nsmallest(top_k_km, "adj_p").set_index("gene_set")
    for gs, row in topk.iterrows():
        df_km = data_cl[[gs, time_col, event_col]].dropna()
        if df_km.shape[0] < min_km_n:
            print(f"  Skip KM {gs}: n={df_km.shape[0]} < {min_km_n}"); continue
        med = df_km[gs].median()
        df_km = df_km.assign(grp=np.where(df_km[gs] >= med, "High", "Low"))

        grp_counts = df_km["grp"].value_counts()
        if (grp_counts.get("High", 0) < min_group_n) or (grp_counts.get("Low", 0) < min_group_n):
            print(f"  Skip KM {gs}: group too small"); continue

        hr   = float(row["HR"])
        fdr  = float(row["adj_p"])
        high_grp = "High" if hr > 1.0 else "Low"
        low_grp  = "Low"  if hr > 1.0 else "High"
        colors   = {high_grp: "#d62728", low_grp: "#1f77b4"}

        kmf = KaplanMeierFitter()
        plt.figure(figsize=(10, 7)); ax = plt.gca()
        for grp in ["High", "Low"]:
            mask = df_km["grp"] == grp
            label_suffix = "high risk" if grp == high_grp else "low risk"
            kmf.fit(df_km.loc[mask, time_col], df_km.loc[mask, event_col],
                    label=f"{grp} (n={mask.sum()}) · {label_suffix}")
            kmf.plot_survival_function(ax=ax, ci_show=True, linewidth=3.2, color=colors[grp])

        # 文本注释（右下角）
        lr = logrank_test(
            df_km[df_km["grp"]=="High"][time_col],
            df_km[df_km["grp"]=="Low"][time_col],
            event_observed_A=df_km[df_km["grp"]=="High"][event_col],
            event_observed_B=df_km[df_km["grp"]=="Low"][event_col]
        )
        pval = float(lr.p_value)
        title_full = shorten_title(gs, 12)

        ax.set_title(f"Cluster {cl} – {title_full}")
        ax.set_xlabel("Time"); ax.set_ylabel("Survival Probability")
        apply_axes_fonts(ax)

        # 图例（左下角）
        style_legend(ax)

        ax.text(0.98, 0.05,
                f"HR={hr:.2f} (per +1 SD) | logrank p={pval:.3g}",
                ha="right", va="bottom", transform=ax.transAxes,
                fontproperties=FP_TNR, fontsize=FS_ANNO,
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"))

        plt.tight_layout()
        fn = os.path.join(out_dir, f"KM_cluster{cl}_{gs}.tiff")
        plt.savefig(fn, dpi=600)
        plt.close()

        km_summaries.append({
            "cluster": cl, "gene_set": gs, "title": title_full,
            "n_total": int(df_km.shape[0]), "n_high": int(grp_counts.get("High", 0)),
            "n_low": int(grp_counts.get("Low", 0)), "cox_HR": hr, "cox_FDR": fdr,
            "logrank_p": pval, "high_risk_group": high_grp
        })

# 6) KM 摘要
if km_summaries:
    pd.DataFrame(km_summaries).sort_values(["cluster","logrank_p"])\
      .to_csv(os.path.join(out_dir, "KM_topk_summary.csv"), index=False)

print("\nAll done! Results in", out_dir)
