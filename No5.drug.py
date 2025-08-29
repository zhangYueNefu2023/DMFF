#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PerCluster_Drug_Repositioning_DSDB_VIS_bold.py

分簇药物重定位（DSigDB + DGIdb）+ 富可视化
- 全局字体：Times New Roman（粗体）
- 所有输出图：TIFF, 600 DPI
"""

import os
import re
import time
import warnings
import requests
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
import gseapy as gp

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

# ================= 全局绘图参数（Times + 粗体 + 大字号 + 600DPI） =================
warnings.filterwarnings("ignore", category=UserWarning)

matplotlib.rcParams.update({
    # 分辨率
    'savefig.dpi': 600,
    'figure.dpi': 600,

    # 字体与粗细
    'font.family': 'Times New Roman',
    'font.serif': ['Times New Roman', 'Times', 'Nimbus Roman No9 L', 'DejaVu Serif', 'serif'],
    'font.weight': 'bold',          # 全局粗体
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',

    # 线宽与外观
    'axes.linewidth': 1.6,
    'xtick.direction': 'out',
    'ytick.direction': 'out',

    # 大小（整体放大）
    'font.size': 26,
    'axes.titlesize': 30,
    'axes.labelsize': 28,
    'xtick.labelsize': 26,
    'ytick.labelsize': 26,
    'legend.fontsize': 26,
    'legend.title_fontsize': 28,

    # 字体嵌入（出版友好）
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

# ======= 工具函数：统一保存 & 轴/色标字体加粗 =======
def safe_savefig(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()

def boldify_axis(ax):
    for lab in (ax.get_xticklabels() + ax.get_yticklabels()):
        lab.set_fontweight('bold')

def boldify_cbar(obj):
    import matplotlib
    if hasattr(obj, 'ax') and isinstance(obj.ax, matplotlib.axes.Axes):
        ax = obj.ax
    elif isinstance(obj, matplotlib.axes.Axes):
        ax = obj
    else:
        return
    if hasattr(ax, 'yaxis') and ax.yaxis is not None:
        if hasattr(ax.yaxis, 'label'):
            ax.yaxis.label.set_fontweight('bold')
        for lab in ax.get_yticklabels():
            lab.set_fontweight('bold')

def clean_dsigdb_term(term: str) -> str:
    """去掉 DSigDB Term 末尾的 CTD 编号及多余括注。"""
    if term is None or (isinstance(term, float) and np.isnan(term)):
        return ""
    name = re.sub(r"\s+CTD\s*\d+.*$", "", str(term))
    name = re.sub(r"\s*\([^)]*\)\s*$", "", name).strip()
    return name

# ========= 稳健的 enrichr 包装：不画官方图，完整取回结果；可选再画 =========
def run_enrichr_safely(glist, gene_sets, organism, outdir, direction):
    """
    返回 DataFrame：Term, P-value, Adjusted P-value, Combined Score, Direction
    - glist 长度 <5 时直接返回空表，避免报错
    - cutoff=1.0，取回完整结果；不画官方图（no_plot=True）
    """
    cols = ["Term", "P-value", "Adjusted P-value", "Combined Score", "Direction"]
    if glist is None or len(glist) < 5:
        return pd.DataFrame(columns=cols)

    try:
        enr = gp.enrichr(
            gene_list=glist,
            gene_sets=gene_sets,
            organism=organism,
            outdir=outdir,
            no_plot=True,   # 不画官方图，避免 cutoff 触发异常
            cutoff=1.0      # 放宽阈值，完整取回
        )
        df = enr.results.copy()
        if df is None or df.empty:
            return pd.DataFrame(columns=cols)
        df["Direction"] = direction
        # 统一列名类型
        if "Adjusted P-value" not in df.columns and "Adjusted P-value " in df.columns:
            df.rename(columns={"Adjusted P-value ": "Adjusted P-value"}, inplace=True)
        return df
    except Exception as e:
        print(f"  ⚠️ Enrichr error ({direction}): {e}")
        return pd.DataFrame(columns=cols)

def maybe_plot_official_enrichr(up, down, gene_sets, organism, outdir):
    """
    若确有显著条目（FDR<0.05），则尝试生成 Enrichr 官方图（继承 Times 粗体样式）。
    """
    # 临时 rc（可省略；全局 rcParams 已设置）
    import matplotlib as mpl
    rc = {
        "font.family": "Times New Roman",
        "font.weight": "bold",
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "savefig.dpi": 600, "pdf.fonttype": 42, "ps.fonttype": 42,
    }
    try:
        # 用 cutoff=0.05，再跑一次“只画图”（没结果时有可能抛错，故 try 包裹）
        if up is not None and len(up) >= 5:
            with mpl.rc_context(rc=rc):
                _ = gp.enrichr(gene_list=up, gene_sets=gene_sets, organism=organism,
                               outdir=outdir, no_plot=False, cutoff=0.05)
        if down is not None and len(down) >= 5:
            with mpl.rc_context(rc=rc):
                _ = gp.enrichr(gene_list=down, gene_sets=gene_sets, organism=organism,
                               outdir=outdir, no_plot=False, cutoff=0.05)
    except Exception as e:
        print(f"  ⚠️ Optional Enrichr plot error: {e}")

# ===================== 运行配置 =====================
cluster_expr = {
    "C0": "cluster_0_RNA.csv",
    "C1": "cluster_1_RNA.csv",
    "C2": "cluster_2_RNA.csv",
}
OUTDIR        = "No5.drug_NrepositioningP"
os.makedirs(OUTDIR, exist_ok=True)

DGIDB_API     = "https://dgidb.org/api/v2/interactions.json"
TOP_N_GENES   = 150     # 上/下 调各取前 N 基因
TOP_DRUGS     = 20      # 每簇展示的 Top 药物数（按 Combined Score 排序）
MAX_NET_DRUGS = 10      # 网络图最多取前 N 药物
MAX_NET_EDGES = 30      # 网络图最多绘制的边数
ENR_LIB       = "DSigDB"  # Enrichr 库
ORGANISM      = "Human"

# ===================== 各类绘图 =====================
def volcano_plot(df_deg, cl, outdir):
    dfv = df_deg.copy()
    dfv["log2FC"] = np.log2(dfv["fc"] + 1e-12)
    dfv["neglog10FDR"] = -np.log10(np.clip(dfv["padj"].values, 1e-300, None))
    sig = (dfv["padj"] < 0.05) & (dfv["log2FC"].abs() >= 1)
    plt.figure(figsize=(7.2, 6.0))
    ax = plt.gca()
    ax.scatter(dfv.loc[~sig, "log2FC"], dfv.loc[~sig, "neglog10FDR"], s=10, alpha=0.45)
    ax.scatter(dfv.loc[sig,  "log2FC"], dfv.loc[sig,  "neglog10FDR"], s=14, alpha=0.85)
    ax.axhline(-np.log10(0.05), linestyle="--", linewidth=1.8)
    ax.axvline( 1, linestyle="--", linewidth=1.8)
    ax.axvline(-1, linestyle="--", linewidth=1.8)
    ax.set_xlabel("log2 Fold Change")
    ax.set_ylabel("-log10 FDR")
    ax.set_title(f"{cl} · DEG Volcano (FDR<0.05, |log2FC|≥1)")
    boldify_axis(ax)
    safe_savefig(os.path.join(outdir, f"{cl}_DEG_volcano.tiff"))

def top_genes_heatmap(expr_dict, df_deg, cl, outdir):
    up25   = df_deg[df_deg["t_stat"] > 0].nlargest(25, "t_stat").index.tolist()
    down25 = df_deg[df_deg["t_stat"] < 0].nsmallest(25, "t_stat").index.tolist()
    genes_heat = [g for g in up25 + down25 if g in expr_dict[cl].index]
    if not genes_heat:
        return
    heat = pd.DataFrame({c: expr_dict[c].loc[genes_heat].mean(axis=1) for c in expr_dict})
    heat = heat.sub(heat.mean(axis=1), axis=0).div(heat.std(axis=1).replace(0, np.nan), axis=0)

    g = sns.clustermap(
        heat, cmap="vlag", col_cluster=False,
        figsize=(7.0, max(6.5, 0.24*len(genes_heat))),
        cbar_kws={"label": "Row z-score"}
    )
    g.ax_heatmap.set_xlabel("Cluster")
    g.ax_heatmap.set_ylabel("")
    g.fig.suptitle(f"{cl} · Top25 Up/Down (row z-score)", y=1.02, fontweight='bold', fontsize=30)
    g.ax_heatmap.tick_params(labelsize=26)
    for lab in (g.ax_heatmap.get_xticklabels() + g.ax_heatmap.get_yticklabels()):
        lab.set_fontweight('bold')
    boldify_cbar(g.cax)
    g.savefig(os.path.join(outdir, f"{cl}_TopGenes_heatmap.tiff"))
    plt.close(g.fig)

def enrichr_bar_and_dot(df_all, cl, outdir, top_n=TOP_DRUGS):
    if df_all.empty:
        return
    dfa = df_all.copy()
    dfa["Drug"] = dfa["Term"].apply(clean_dsigdb_term)
    dfa = dfa.sort_values("Combined Score", ascending=False).head(top_n)

    # 条形图（按 Combined Score）
    plt.figure(figsize=(7.6, 0.5*len(dfa)))
    ax = sns.barplot(data=dfa, y="Drug", x="Combined Score", hue="Direction", dodge=False)
    ax.set_xlabel("Combined score")
    ax.set_ylabel("")
    ax.set_title(f"{cl} · DSigDB Top {len(dfa)} (Combined score)")
    boldify_axis(ax)
    safe_savefig(os.path.join(outdir, f"{cl}_DSigDB_top{len(dfa)}_bar.tiff"))

    # 点图（-log10 FDR vs Combined Score）
    if "Adjusted P-value" in dfa.columns:
        dfa["neglog10FDR"] = -np.log10(np.clip(pd.to_numeric(dfa["Adjusted P-value"], errors="coerce").values, 1e-300, None))
        plt.figure(figsize=(7.6, 0.5*len(dfa)))
        ax = sns.scatterplot(
            data=dfa, y="Drug", x="neglog10FDR",
            size="Combined Score", hue="Direction", legend=True
        )
        ax.set_xlabel("-log10(FDR)")
        ax.set_ylabel("")
        ax.set_title(f"{cl} · DSigDB Top {len(dfa)} (-log10 FDR, size=Combined score)")
        boldify_axis(ax)
        safe_savefig(os.path.join(outdir, f"{cl}_DSigDB_top{len(dfa)}_dot.tiff"))

def dgidb_network(df_interactions, cl, outdir,
                  max_drugs=MAX_NET_DRUGS, max_edges=MAX_NET_EDGES):
    if df_interactions is None or df_interactions.empty:
        return
    edge_df = df_interactions.copy()
    edge_df = edge_df.dropna(subset=["Drug", "TargetGene"])
    drug_counts = edge_df.groupby("Drug").size().sort_values(ascending=False)
    keep_drugs = set(drug_counts.head(max_drugs).index)
    edge_df = edge_df[edge_df["Drug"].isin(keep_drugs)].head(max_edges)
    if edge_df.empty:
        return

    G = nx.Graph()
    drug_nodes = sorted(set(edge_df["Drug"]))
    gene_nodes = sorted(set(edge_df["TargetGene"]))
    G.add_nodes_from([(d, {"bipartite": 0}) for d in drug_nodes])
    G.add_nodes_from([(g, {"bipartite": 1}) for g in gene_nodes])
    for _, row in edge_df.iterrows():
        G.add_edge(row["Drug"], row["TargetGene"])

    pos = {}
    left_x, right_x = 0.0, 1.0
    for i, d in enumerate(drug_nodes):
        pos[d] = (left_x, i / max(1, len(drug_nodes)-1))
    for j, g in enumerate(gene_nodes):
        pos[g] = (right_x, j / max(1, len(gene_nodes)-1))

    plt.figure(figsize=(9, max(6.5, 0.28*(len(drug_nodes)+len(gene_nodes)))))
    nx.draw_networkx_edges(G, pos, alpha=0.5, width=1.8)
    nx.draw_networkx_nodes(G, pos, nodelist=drug_nodes, node_shape='s', node_size=740)
    nx.draw_networkx_nodes(G, pos, nodelist=gene_nodes,  node_shape='o', node_size=420)
    nx.draw_networkx_labels(G, pos, font_size=16, font_family='Times New Roman', font_weight='bold')
    plt.axis("off")
    plt.title(f"{cl} · DGIdb Drug–Target Network (≤{max_drugs} drugs, ≤{max_edges} edges)")
    safe_savefig(os.path.join(outdir, f"{cl}_DGIdb_network.tiff"))

# ===================== 1) 读取并去重表达矩阵 =====================
expr = {}
for cl, fn in cluster_expr.items():
    df = pd.read_csv(fn, index_col=0)
    df = df.groupby(df.index).mean()
    expr[cl] = df
    print(f"Loaded {cl}: {df.shape[0]} unique genes × {df.shape[1]} samples")

# 统一基因集合
common_genes = set.intersection(*[set(df.index) for df in expr.values()])
common_genes = sorted(list(common_genes))
if len(common_genes) == 0:
    raise ValueError("三个簇之间没有共同基因，请检查输入矩阵。")
print(f"Common genes across clusters: {len(common_genes)}")

# ===================== 2) 差异表达：簇 vs 其余 =====================
degs = {}
for cl, df_cl in expr.items():
    others = pd.concat([expr[c] for c in expr if c != cl], axis=1)
    pvals, tstats = [], []
    x_mat = df_cl.loc[common_genes]
    y_mat = others.loc[common_genes]
    for i, g in enumerate(common_genes):
        x = x_mat.iloc[i].dropna().values
        y = y_mat.iloc[i].dropna().values
        if len(x) < 2 or len(y) < 2:
            t, p = np.nan, np.nan
        else:
            t, p = ttest_ind(x, y, equal_var=False, nan_policy="omit")
        tstats.append(t); pvals.append(p)

    df = pd.DataFrame({"gene": common_genes, "t_stat": tstats, "pval": pvals}).set_index("gene")
    df["padj"] = multipletests(df["pval"].fillna(1), method="fdr_bh")[1]
    df["fc"]   = x_mat.mean(axis=1) / (y_mat.mean(axis=1) + 1e-8)
    df = df.sort_values("padj")
    path = os.path.join(OUTDIR, f"{cl}_DEG.csv")
    df.to_csv(path)
    degs[cl] = df
    print(f"✔ {cl}_DEG.csv saved ({df.shape[0]} genes)")

# ===================== 3) Enrichr DSigDB + 自定义可视化 + DGIdb =====================
drug_matrix_collect = {}  # {drug_name: {cluster: -log10FDR_best}}
drug_tables = []

for cl, df_deg in degs.items():
    print(f"\n=== Cluster {cl} ===")
    subdir = os.path.join(OUTDIR, cl)
    os.makedirs(subdir, exist_ok=True)

    # ------- 构建签名 -------
    sig = df_deg[df_deg["padj"] < 0.05]
    if sig["t_stat"].gt(0).sum() < 5 or sig["t_stat"].lt(0).sum() < 5:
        print("  padj<0.05 基因不足，退用全体 DEG")
        sig = df_deg
    up   = sig[sig["t_stat"] > 0].nlargest(TOP_N_GENES, "t_stat").index.tolist()
    down = sig[sig["t_stat"] < 0].nsmallest(TOP_N_GENES, "t_stat").index.tolist()
    print(f"  Signature: {len(up)} up / {len(down)} down genes")

    # ------- Volcano + Heatmap -------
    volcano_plot(df_deg, cl, subdir)
    top_genes_heatmap(expr, df_deg, cl, subdir)

    # ------- Enrichr（稳健模式：不画官方图，完整结果；必要时再画） -------
    df_up = run_enrichr_safely(up,   ENR_LIB, ORGANISM, subdir, "Up")
    df_dn = run_enrichr_safely(down, ENR_LIB, ORGANISM, subdir, "Down")
    df_all = pd.concat([df_up, df_dn], axis=0, ignore_index=True)

    enrichr_fn = os.path.join(subdir, f"{cl}_DSigDB_enrichr.csv")
    df_all.to_csv(enrichr_fn, index=False)
    print(f"  Enrichr → {os.path.basename(enrichr_fn)} ({len(df_all)} rows)")

    # 若有显著条目（FDR<0.05），可选生成官方 PDF/PNG（继承 Times 粗体）
    if not df_all.empty and (pd.to_numeric(df_all["Adjusted P-value"], errors="coerce") < 0.05).any():
        maybe_plot_official_enrichr(up, down, ENR_LIB, ORGANISM, subdir)

    # ------- 自定义 Top 药物可视化 -------
    enrichr_bar_and_dot(df_all, cl, subdir, top_n=TOP_DRUGS)

    # ------- 解析药名（去 CTD 编号），并记录用于跨簇热图 -------
    if not df_all.empty and "Adjusted P-value" in df_all.columns:
        df_all["DrugName"] = df_all["Term"].apply(clean_dsigdb_term)
        best = (df_all
                .groupby("DrugName", as_index=False)["Adjusted P-value"]
                .min())
        best = best.sort_values("Adjusted P-value").head(max(TOP_DRUGS, 50))
        best["neglog10FDR"] = -np.log10(np.clip(pd.to_numeric(best["Adjusted P-value"], errors="coerce").values, 1e-300, None))
        for _, r in best.iterrows():
            drug = r["DrugName"]
            drug_matrix_collect.setdefault(drug, {})
            drug_matrix_collect[drug][cl] = r["neglog10FDR"]

        tbl = df_all.copy()
        cols_keep = ["DrugName","Direction","Adjusted P-value","Combined Score","Overlap","P-value","Term"]
        existing = [c for c in cols_keep if c in tbl.columns]
        tbl = tbl[existing].sort_values(["Adjusted P-value","Combined Score"], ascending=[True, False]).head(TOP_DRUGS)
        tbl.insert(0, "Cluster", cl)
        tbl.rename(columns={"Adjusted P-value":"FDR"}, inplace=True)
        tbl.to_csv(os.path.join(subdir, f"{cl}_Top{TOP_DRUGS}_drugs.csv"), index=False)
        drug_tables.append(tbl)

    # ------- DGIdb 验证并绘图 -------
    drug_names = []
    if "DrugName" in df_all.columns:
        drug_names = df_all["DrugName"].dropna().unique().tolist()
        drug_names = [d for d in drug_names if d][:TOP_DRUGS]
    print(f"  Validating {len(drug_names)} drugs via DGIdb …")

    interactions = []
    genes_query = ",".join(up + down)
    for drug in drug_names:
        try:
            resp = requests.get(DGIDB_API, params={"genes": genes_query, "drugs": drug}, timeout=30)
            if resp.status_code != 200:
                print(f"    DGIdb Error {resp.status_code} for {drug}")
                continue
            data = resp.json()
        except Exception as e:
            print(f"    DGIdb exception for {drug}: {e}")
            continue

        # 新版 DGIdb 返回结构可能不同，这里兼容 matchedTerms / interactions
        matched = data.get("matchedTerms", [])
        for term in matched:
            gene = term.get("geneName")
            for inter in term.get("interactions", []):
                interactions.append({
                    "Drug": drug,
                    "TargetGene": gene,
                    "Interaction": inter.get("interactionType")
                })
        time.sleep(0.2)  # 轻微节流

    df_int = pd.DataFrame(interactions, columns=["Drug", "TargetGene", "Interaction"])
    dg_fn = os.path.join(subdir, f"{cl}_Drug_Target_Interactions.csv")
    df_int.to_csv(dg_fn, index=False)
    print(f"  DGIdb → {os.path.basename(dg_fn)} ({len(df_int)} rows)")
    dgidb_network(df_int, cl, subdir, max_drugs=MAX_NET_DRUGS, max_edges=MAX_NET_EDGES)

# ===================== 4) 跨簇药物热图（-log10 FDR） =====================
if drug_matrix_collect:
    mat = pd.DataFrame(drug_matrix_collect).T  # rows: drug, cols: cluster
    mat = mat.loc[mat.notna().any(axis=1)]
    top_rows = min(60, mat.shape[0])
    mat = mat.sort_values(mat.columns.tolist(), ascending=False).head(top_rows)

    plt.figure(figsize=(7.2, max(6.5, 0.28*mat.shape[0])))
    ax = sns.heatmap(mat.fillna(0.0), cmap="rocket_r", cbar_kws={"label": "-log10(FDR)"})
    ax.set_xlabel("Cluster"); ax.set_ylabel("Drug")
    ax.set_title("Cross-cluster drug significance (best -log10 FDR)")
    boldify_axis(ax)
    cbar = ax.collections[0].colorbar
    boldify_cbar(cbar)
    safe_savefig(os.path.join(OUTDIR, "CrossCluster_TopDrugs_heatmap.tiff"))

# ===================== 5) 导出汇总总表 =====================
if drug_tables:
    all_tbl = pd.concat(drug_tables, axis=0, ignore_index=True)
    all_tbl.to_csv(os.path.join(OUTDIR, f"AllClusters_Top{TOP_DRUGS}_drugs_summary.csv"), index=False)

print("\n✅ Drug repositioning via DSigDB + DGIdb with robust Enrichr (Times New Roman bold, TIFF@600dpi) complete.")
