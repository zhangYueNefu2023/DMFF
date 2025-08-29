#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import pandas as pd
import numpy as np
from scipy.stats import zscore, ttest_ind, kruskal
from statsmodels.stats.multitest import multipletests
import seaborn as sns
import matplotlib.pyplot as plt
from gseapy.parser import get_library
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
# from umap import UMAP  # 如果安装了 umap-learn

# ============ 参数配置 ============
rna_pattern = "cluster_*_RNA.csv"
out_dir     = "No1.gsva_full_pipeline"
os.makedirs(out_dir, exist_ok=True)

# Enrichr KEGG_2016 基因集
lib_name = "KEGG_2016"
organism = "Human"
min_size = 5
max_size = 2000

# ============ 1. 合并表达矩阵 & 记录 Cluster ============
dfs = []
cluster_map = {}
for fn in sorted(glob.glob(rna_pattern)):
    cid = int(os.path.basename(fn).split("_")[1])
    df  = pd.read_csv(fn, index_col=0).apply(pd.to_numeric, errors='coerce')
    dfs.append(df)
    for s in df.columns:
        cluster_map[s] = cid
expr_df = pd.concat(dfs, axis=1)
print("1. Loaded expression:", expr_df.shape)

# ============ 2. 下载通路 & 计算活性评分 ============
pathways = get_library(lib_name, organism, min_size, max_size)
print("2. Downloaded pathways:", len(pathways))

expr_z = expr_df.T.apply(zscore, axis=1, result_type="broadcast").T

ps = pd.DataFrame(index=pathways.keys(), columns=expr_z.columns, dtype=float)
for pname, genes in pathways.items():
    valid = [g for g in genes if g in expr_z.index]
    if len(valid) < min_size:
        continue
    ps.loc[pname] = expr_z.loc[valid].mean(axis=0)
ps = ps.dropna(how="any")
ps.to_csv(os.path.join(out_dir, "pathway_scores.csv"))
print("3. Computed pathway scores:", ps.shape)

# ============ 3. 合并 Cluster 标签 ============
scores_T = ps.T
scores_T["Cluster"] = scores_T.index.map(cluster_map)
scores_T.to_csv(os.path.join(out_dir, "scores_with_clusters.csv"))
print("4. Saved scores_with_clusters.csv")

# ============ 4. 差异通路初筛 (t-test) ============
de_dir = os.path.join(out_dir, "DE_pathways")
os.makedirs(de_dir, exist_ok=True)
for k in sorted(scores_T["Cluster"].unique()):
    grp1 = scores_T[scores_T["Cluster"] == k].drop("Cluster", axis=1)
    grp2 = scores_T[scores_T["Cluster"] != k].drop("Cluster", axis=1)
    recs = []
    for pw in grp1.columns:
        a = grp1[pw].dropna()
        b = grp2[pw].dropna()
        if len(a) < 2 or len(b) < 2:
            continue
        stat, p = ttest_ind(a, b, equal_var=False, nan_policy="omit")
        fc = a.mean() / (b.mean() + 1e-8)
        recs.append((pw, p, fc))
    dfde = pd.DataFrame(recs, columns=["pathway", "p_value", "fold_change"])
    dfde["adj_p"] = multipletests(dfde["p_value"], method="fdr_bh")[1]
    dfde.sort_values("adj_p") \
         .to_csv(os.path.join(de_dir, f"DE_pathways_cluster_{k}.csv"), index=False)
print("5. Saved DE_pathways for each cluster")

# ============ 5. 选 TopN 差异通路 & Kruskal-Wallis ============
cluster_means = scores_T.groupby("Cluster").mean().T
cluster_means["max_diff"] = cluster_means.max(axis=1) - cluster_means.min(axis=1)
topN = 20
top_pathways = cluster_means.nlargest(topN, "max_diff").index.tolist()

pvals = []
for pw in top_pathways:
    groups = [scores_T[scores_T["Cluster"] == k][pw].dropna()
              for k in sorted(scores_T["Cluster"].unique())]
    if any(len(g) < 2 for g in groups):
        pvals.append(1.0)
    else:
        _, p = kruskal(*groups)
        pvals.append(p)
df_pw = pd.DataFrame({"pathway": top_pathways, "p": pvals})
df_pw["adj_p"] = multipletests(df_pw["p"], method="fdr_bh")[1]
sig_paths = df_pw[df_pw["adj_p"] < 0.05]["pathway"].tolist()
print(f"6. Significant (adj_p<0.05): {len(sig_paths)} pathways")

# ============ 6. 箱线图 (仅显著通路) ============

# 去掉通路名里的 hsa 编号
clean_names = {p: p.split(" Homo")[0] if " Homo" in p else p.split(" hsa")[0] for p in sig_paths}
melt = scores_T[sig_paths + ["Cluster"]].rename(columns=clean_names) \
    .melt(id_vars="Cluster", var_name="pathway", value_name="score")

# A4 横向基础尺寸 + 按通路数量自适应加宽
A4_W, A4_H = 11.69, 8.27
per_cat_w  = 0.5  # 每个通路最小分配宽度
fig_w      = max(A4_W, per_cat_w * max(1, len(melt["pathway"].unique())))
fig_h      = 8.0   # 高度再大一点

plt.rcParams["font.family"] = "Times New Roman"  # 全局字体
plt.rcParams["font.weight"] = "bold"             # 全局加粗

fig, ax = plt.subplots(figsize=(fig_w, fig_h))

sns.boxplot(
    x="pathway", y="score", hue="Cluster", data=melt,
    palette="Set2",
    fliersize=0, linewidth=1.2
)

# 去掉图例
ax.get_legend().remove()

# 字体大、加粗
plt.xticks(rotation=45, ha="right", fontsize=20, fontweight="bold")
plt.yticks(fontsize=20, fontweight="bold")

# 坐标轴标签
ax.set_xlabel("Pathway", fontsize=25, fontweight="bold", labelpad=10)
ax.set_ylabel("Score", fontsize=25, fontweight="bold", labelpad=10)

# 去掉标题
ax.set_title("", fontsize=0)

# 紧凑布局
plt.tight_layout()
plt.savefig(
    os.path.join(out_dir, "significant_pathways_boxplot.tiff"),
    dpi=600, format="tiff", bbox_inches="tight"
)
plt.close()


# ============ 7. PCA & t-SNE 可视化 ============
X = scores_T.drop("Cluster", axis=1).values
y = scores_T["Cluster"].values

# PCA
pca = PCA(n_components=2, random_state=42)
pcs = pca.fit_transform(X)
plt.figure(figsize=(6,5))
sns.scatterplot(x=pcs[:,0], y=pcs[:,1], hue=y, palette="Set2", s=40, alpha=0.8)
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
plt.legend(title="Cluster")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "pca_pathway_scores.tiff"), dpi=600)
plt.close()

# t-SNE
tsne = TSNE(n_components=2, random_state=42, learning_rate="auto", init="pca")
ts = tsne.fit_transform(X)
plt.figure(figsize=(6,5))
sns.scatterplot(x=ts[:,0], y=ts[:,1], hue=y, palette="Set2", s=40, alpha=0.8)
plt.xlabel("t-SNE1")
plt.ylabel("t-SNE2")
plt.legend(title="Cluster")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "tsne_pathway_scores.tiff"), dpi=600)
plt.close()

# （可选）UMAP
# umap = UMAP(n_components=2, random_state=42)
# um = umap.fit_transform(X)
# plt.figure(figsize=(6,5))
# sns.scatterplot(x=um[:,0], y=um[:,1], hue=y, palette="Set2", s=40, alpha=0.8)
# plt.xlabel("UMAP1")
# plt.ylabel("UMAP2")
# plt.legend(title="Cluster")
# plt.tight_layout()
# plt.savefig(os.path.join(out_dir, "umap_pathway_scores.png"), dpi=300)
# plt.close()

print("✅ Full pipeline complete, results in:", out_dir)
