import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from lifelines.utils import concordance_index
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import shap
import random
import networkx as nx

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

plt.rcParams['font.family'] = 'Times New Roman'
sns.set_theme(style="whitegrid", context="paper", font_scale=1.3)
contrast_palette = sns.color_palette("Set2", n_colors=8)

########################################
# 0. 可视化蛋白质网络函数
########################################
def visualize_protein_network(edge_index, n_nodes):
    edges = edge_index.cpu().numpy().T
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    for (u, v) in edges:
        if u != v:
            G.add_edge(u, v)
    plt.figure(figsize=(12, 12), dpi=600)
    pos = nx.spring_layout(G, k=0.2, iterations=100, seed=42)
    nx.draw_networkx_nodes(G, pos, node_size=50, node_color=contrast_palette[0], alpha=0.8)
    nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.3, width=0.5)
    plt.title("Protein PPI Network Visualization", loc='center', fontsize=18)
    plt.axis('off')
    plt.show()

########################################
# 1. 数据集构建
########################################
class OmicsSurvivalDataset(Dataset):
    def __init__(self, mirna_file, protein_file, star_tpm_file, clinical_file, survival_file):
        self.df_mirna = pd.read_csv(mirna_file, sep='\t', index_col=0).T
        self.df_protein = pd.read_csv(protein_file, sep='\t', index_col=0).T
        self.df_star_tpm = pd.read_csv(star_tpm_file, sep='\t', index_col=0).T

        df_clinical = pd.read_csv(clinical_file, sep='\t', index_col=0)
        df_survival = pd.read_csv(survival_file, sep='\t', index_col=0)

        self.df_mirna.index = self.df_mirna.index.astype(str).str.strip()
        self.df_protein.index = self.df_protein.index.astype(str).str.strip()
        self.df_star_tpm.index = self.df_star_tpm.index.astype(str).str.strip()
        df_clinical.index = df_clinical.index.astype(str).str.strip()
        df_survival.index = df_survival.index.astype(str).str.strip()

        self.df_surv = pd.merge(df_clinical, df_survival, left_index=True, right_index=True, how="inner")
        common_ids = set(self.df_mirna.index) & set(self.df_protein.index) & set(self.df_star_tpm.index) & set(self.df_surv.index)
        common_ids = list(common_ids)
        common_ids.sort()

        self.df_mirna = self.df_mirna.loc[common_ids]
        self.df_protein = self.df_protein.loc[common_ids]
        self.df_star_tpm = self.df_star_tpm.loc[common_ids]
        self.df_surv = self.df_surv.loc[common_ids]
        self.sample_ids = common_ids

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, index):
        sample_id = self.sample_ids[index]
        mirna = torch.tensor(self.df_mirna.loc[sample_id].values, dtype=torch.float32)
        protein = torch.tensor(self.df_protein.loc[sample_id].values, dtype=torch.float32)
        star_tpm = torch.tensor(self.df_star_tpm.loc[sample_id].values, dtype=torch.float32)
        duration = torch.tensor(self.df_surv.loc[sample_id, 'OS.time'], dtype=torch.float32)
        event = torch.tensor(self.df_surv.loc[sample_id, 'OS'], dtype=torch.float32)
        return {'mirna': mirna, 'protein': protein, 'star_tpm': star_tpm,
                'duration': duration, 'event': event}

########################################
# 2. 模型构建
########################################
class OmicsAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(OmicsAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(),
            nn.Linear(512, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.LeakyReLU(),
            nn.Linear(512, input_dim)
        )
    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z

class SurvivalHead(nn.Module):
    def __init__(self, latent_dim):
        super(SurvivalHead, self).__init__()

        # 增加更多的隐藏层（MLP结构）
        self.fc1 = nn.Linear(latent_dim, 512)  # 第一个隐藏层
        self.fc2 = nn.Linear(512, 256)  # 第二个隐藏层
        self.fc3 = nn.Linear(256, 128)  # 第三个隐藏层
        self.fc4 = nn.Linear(128, 1)  # 输出层

        # 激活函数
        self.relu = nn.ReLU()

        # Dropout正则化
        self.dropout = nn.Dropout(p=0.5)

        # Batch Normalization
        self.batch_norm1 = nn.BatchNorm1d(512)
        self.batch_norm2 = nn.BatchNorm1d(256)
        self.batch_norm3 = nn.BatchNorm1d(128)

    def forward(self, x):
        # 通过各个隐藏层并加入非线性激活
        x = self.fc1(x)
        x = self.batch_norm1(x)  # 批归一化
        x = self.relu(x)
        x = self.dropout(x)  # Dropout正则化

        x = self.fc2(x)
        x = self.batch_norm2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        x = self.batch_norm3(x)
        x = self.relu(x)
        x = self.dropout(x)

        # 最后通过输出层
        risk = self.fc4(x)

        return risk

class FusionLayer(nn.Module):
    def __init__(self, latent_dim, num_modalities):
        super(FusionLayer, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 1)
        )
    def forward(self, features_list):
        attn_weights = []
        for feat in features_list:
            attn = self.attention(feat)
            attn_weights.append(attn)
        attn_weights = torch.stack(attn_weights, dim=1)  # (batch, modalities, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)
        features = torch.stack(features_list, dim=1)      # (batch, modalities, latent_dim)
        fused_feature = torch.sum(attn_weights * features, dim=1)
        return fused_feature, attn_weights

try:
    from torch_geometric.nn import GCNConv, global_mean_pool
except ImportError:
    raise ImportError("请安装 torch_geometric 库以支持 GNN 部分。")

class ProteinBranchGNN(nn.Module):
    def __init__(self, n_nodes, latent_dim):
        super(ProteinBranchGNN, self).__init__()
        self.linear = nn.Linear(1, latent_dim)
        self.conv1 = GCNConv(latent_dim, latent_dim)
        self.conv2 = GCNConv(latent_dim, latent_dim)
        self.n_nodes = n_nodes
    def forward(self, x, edge_index):
        batch_size = x.size(0)
        x = x.view(-1, 1)
        x = self.linear(x)
        batch_tensor = torch.arange(batch_size, device=x.device).repeat_interleave(self.n_nodes)
        x = torch.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch_tensor)
        return x

class MultiModalSurvivalModel_GNN(nn.Module):
    def __init__(self, input_dims, latent_dim, n_protein_nodes):
        super(MultiModalSurvivalModel_GNN, self).__init__()
        self.mirna_encoder = nn.Sequential(
            nn.Linear(input_dims['mirna'], 512),
            nn.LeakyReLU(),
            nn.Linear(512, latent_dim)
        )
        self.star_tpm_encoder = nn.Sequential(
            nn.Linear(input_dims['star_tpm'], 512),
            nn.LeakyReLU(),
            nn.Linear(512, latent_dim)
        )
        self.protein_gnn = ProteinBranchGNN(n_nodes=n_protein_nodes, latent_dim=latent_dim)
        self.fusion_layer = FusionLayer(latent_dim, num_modalities=3)
        self.survival_head = SurvivalHead(latent_dim)
    def forward(self, inputs, protein_edge_index):
        mirna_feat = self.mirna_encoder(inputs['mirna'])
        star_tpm_feat = self.star_tpm_encoder(inputs['star_tpm'])
        protein_feat = self.protein_gnn(inputs['protein'], protein_edge_index)
        features = [mirna_feat, protein_feat, star_tpm_feat]
        fused_feature, attn_weights = self.fusion_layer(features)
        risk = self.survival_head(fused_feature)
        return risk, fused_feature, attn_weights

########################################
# 3. 损失函数与训练策略
########################################
def cox_loss(risk, durations, events):
    risk = risk.squeeze()
    risk = torch.clamp(risk, min=-10, max=10)
    order = torch.argsort(durations, descending=True)
    sorted_risk = risk[order]
    sorted_events = events[order]
    exp_risk = torch.exp(sorted_risk)
    cum_sum_exp = torch.cumsum(exp_risk, dim=0)
    log_likelihood = sorted_risk - torch.log(cum_sum_exp + 1e-8)
    loss = -torch.sum(log_likelihood * sorted_events) / (torch.sum(sorted_events) + 1e-8)
    return loss

def attention_kl_loss(attn_weights):
    attn = attn_weights.squeeze(-1)
    num_modalities = attn.shape[1]
    uniform = torch.full_like(attn, 1.0 / num_modalities)
    kl_div = torch.sum(uniform * torch.log((uniform + 1e-8) / (attn + 1e-8)), dim=1)
    return torch.mean(kl_div)

########################################
# 4. 预训练与联合训练函数
########################################
def apply_mask(x, mask_ratio=0.2):
    mask = (torch.rand(x.shape) > mask_ratio).float().to(x.device)
    return x * mask

def pretrain_autoencoder(autoencoder, modality, dataloader, device, num_epochs=5, lr=1e-3):
    autoencoder.to(device)
    optimizer = optim.Adam(autoencoder.parameters(), lr=lr)
    losses = []
    autoencoder.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            x = batch[modality].to(device)
            masked_x = apply_mask(x, mask_ratio=0.2)
            recon, _ = autoencoder(masked_x)
            loss = nn.MSELoss()(recon, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * x.size(0)
        avg_loss = epoch_loss / len(dataloader.dataset)
        losses.append(avg_loss)
        print(f"[Pretrain-{modality}] Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    return losses

def train_joint(model, dataloader, optimizer, num_epochs=10, device='cpu', lambda_attn=0.1, protein_edge_index=None):
    model.to(device)
    model.train()
    epoch_losses = []
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            inputs = {
                'mirna': batch['mirna'].to(device),
                'star_tpm': batch['star_tpm'].to(device),
                'protein': batch['protein'].to(device)
            }
            durations = batch['duration'].to(device)
            events = batch['event'].to(device)
            optimizer.zero_grad()
            risk, _, attn_weights = model(inputs, protein_edge_index)
            base_loss = cox_loss(risk, durations, events)
            reg_loss = attention_kl_loss(attn_weights)
            loss = base_loss + lambda_attn * reg_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_loss += loss.item() * batch['mirna'].size(0)
        avg_loss = epoch_loss / len(dataloader.dataset)
        print(f"[Joint Train] Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
        epoch_losses.append(avg_loss)
    return epoch_losses

########################################
# 5. 构建 STRING 蛋白质网络
########################################
import numpy as np
import torch

def build_protein_edge_index(string_file, protein_file, score_threshold=0.95):
    # 读取蛋白质数据
    df_protein = pd.read_csv(protein_file, sep='\t', index_col=0).T
    protein_names = list(df_protein.columns)
    protein_to_idx = {protein: idx for idx, protein in enumerate(protein_names)}

    # 读取STRING数据
    df_string = pd.read_csv(string_file, sep='\t')

    # 筛选符合阈值的蛋白质交互数据
    df_string_filtered = df_string[df_string['combined_score'] >= score_threshold].copy()
    print(f"Filtered edges count: {len(df_string_filtered)}")  # 输出筛选后的边数

    if len(df_string_filtered) == 0:
        print("No edges found after filtering with the given score threshold!")
        return torch.tensor([[], []], dtype=torch.long)  # 如果没有边，返回空的edge_index

    # 映射蛋白质名到索引
    def map_protein(prot):
        return protein_to_idx.get(prot, None)

    df_string_filtered['idx1'] = df_string_filtered['node1'].apply(map_protein)
    df_string_filtered['idx2'] = df_string_filtered['node2'].apply(map_protein)

    # 去掉没有匹配到的蛋白质数据
    df_string_filtered = df_string_filtered.dropna(subset=['idx1', 'idx2'])
    print(f"Valid edges count after removing unmapped proteins: {len(df_string_filtered)}")

    # 确保索引是整数
    df_string_filtered['idx1'] = df_string_filtered['idx1'].astype(int)
    df_string_filtered['idx2'] = df_string_filtered['idx2'].astype(int)

    # 构建 edge_index
    edge_index = torch.tensor(df_string_filtered[['idx1', 'idx2']].values.T, dtype=torch.long)

    # 对边进行排序并去重
    sorted_edges = np.sort(edge_index.cpu().numpy(), axis=0)
    sorted_edges = np.unique(sorted_edges, axis=1)
    edge_index = torch.tensor(sorted_edges, dtype=torch.long)

    print(f"Final edge_index shape: {edge_index.shape}")  # 输出最终的 edge_index 形状
    return edge_index

########################################
# 6. 单独亚型热图函数：保留横坐标（融合特征名称）但只显示部分标签
########################################

def plot_subtype_heatmaps(fusion_features_df, k, output_dir="subtype_heatmaps"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cluster_values = sorted(fusion_features_df["Cluster"].unique())

    # 聚类特定热图生成（同之前的代码逻辑）
    for cluster_id in cluster_values:
        cluster_data = fusion_features_df[fusion_features_df["Cluster"] == cluster_id].drop("Cluster", axis=1)
        if cluster_data.empty:
            continue
        plt.figure(figsize=(10, 6), dpi=600)

        ax = sns.heatmap(cluster_data, cmap="RdBu_r", cbar=True,
                         xticklabels=True,  # 显示列名称
                         yticklabels=True)  # 显示行名称

        # 去除坐标轴的刻度
        ax.set_xticks([])  # 去除横坐标刻度
        ax.set_yticks([])  # 去除纵坐标刻度

        plt.title(f"Subtype Heatmap: Cluster {cluster_id} (k={k})", loc='center', fontsize=14)
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        heatmap_file = os.path.join(output_dir, f"cluster_{cluster_id}_heatmap_k_{k}.png")
        plt.tight_layout()
        plt.savefig(heatmap_file, dpi=600, bbox_inches='tight')
        plt.close()
        print(f"[Subtype Heatmap] Cluster={cluster_id}, saved to {heatmap_file}")

    # 融合特征热图（隐藏底部特征标签，并显示聚类样本）
    plt.figure(figsize=(12, 8), dpi=600)
    fusion_data = fusion_features_df.drop("Cluster", axis=1)

    # 聚类标签在左侧显示样本
    fusion_data['Cluster'] = fusion_features_df['Cluster']
    fusion_data = fusion_data.set_index('Cluster')  # 设置聚类标签为行索引

    ax = sns.heatmap(fusion_data, cmap="RdBu_r", cbar=True,
                     xticklabels=False,  # 隐藏底部的特征标签
                     yticklabels=True)  # 显示左边的样本聚类标签

    # 去除坐标轴的刻度
    ax.set_xticks([])  # 去除横坐标刻度
    ax.set_yticks([])  # 去除纵坐标刻度

    plt.title("Fusion Features Heatmap", loc='center', fontsize=18)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    fusion_heatmap_file = os.path.join(output_dir, "fusion_features_heatmap.png")
    plt.tight_layout()
    plt.savefig(fusion_heatmap_file, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"[Fusion Heatmap] saved to {fusion_heatmap_file}")


########################################
# 7. 保存聚类指标与生存分析结果，并绘制折线图（2行布局，下行2个图拉宽居中）
########################################

import os
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
from sklearn.cluster import KMeans
import pandas as pd
from itertools import combinations

def save_clustering_and_survival_results(fusion_features, data_df, sample_ids, cluster_numbers,
                                         output_dir_metrics="clustering_metrics",
                                         output_dir_survival="survival_analysis"):

    overall_metrics = {
        "k": [],
        "concordance": [],
        "partial_aic": [],
        "likelihood_ratio": [],
        "silhouette": [],
        "calinski_harabasz": []
    }

    sns.set_palette("Set2")

    for k in cluster_numbers:
        k_metric_dir = os.path.join(output_dir_metrics, f"k_{k}")
        if not os.path.exists(k_metric_dir):
            os.makedirs(k_metric_dir)
        k_survival_dir = os.path.join(output_dir_survival, f"k_{k}")
        if not os.path.exists(k_survival_dir):
            os.makedirs(k_survival_dir)

        # 聚类
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(fusion_features)

        # 计算聚类评估指标
        sil_score = silhouette_score(fusion_features, cluster_labels)
        ch_score = calinski_harabasz_score(fusion_features, cluster_labels)

        # 使用融合特征进行生存分析（直接使用所有特征）
        # 假设 fusion_features 是一个二维数组（每个样本有多个特征）
        fusion_features_df = pd.DataFrame(fusion_features)  # 将融合特征转为DataFrame

        # 将所有融合特征添加到df_cox中
        df_cox = data_df[["OS.time", "OS"]].copy()
        # 这里我们直接使用所有特征，假设fusion_features是二维数组，每一列代表一个特征
        for i in range(fusion_features_df.shape[1]):
            df_cox[f"Feature_{i}"] = fusion_features_df.iloc[:, i]  # 将每个特征加入到DataFrame中

        # 创建Cox比例风险模型
        cph = CoxPHFitter(penalizer=1)
        cph.fit(df_cox, duration_col="OS.time", event_col="OS", show_progress=True)

        # 获取Cox回归结果
        concordance = cph.concordance_index_
        partial_aic = cph.AIC_partial_
        lrt = cph.log_likelihood_ratio_test()
        lrt_pvalue = lrt.p_value

        # 保存当前k值下的各项指标
        overall_metrics["k"].append(k)
        overall_metrics["concordance"].append(concordance)
        overall_metrics["partial_aic"].append(partial_aic)
        overall_metrics["likelihood_ratio"].append(lrt_pvalue)
        overall_metrics["silhouette"].append(sil_score)
        overall_metrics["calinski_harabasz"].append(ch_score)

        # 保存聚类指标和生存分析结果
        metrics_file = os.path.join(k_metric_dir, "clustering_metrics.txt")
        with open(metrics_file, "w") as f:
            f.write(f"k: {k}\n")
            f.write(f"Concordance Index: {concordance:.4f}\n")
            f.write(f"Partial AIC: {partial_aic:.4f}\n")
            f.write(f"Likelihood Ratio Test p-value: {lrt_pvalue:.4f}\n")
            f.write(f"Silhouette Score: {sil_score:.4f}\n")
            f.write(f"Calinski Harabasz Index: {ch_score:.4f}\n")
        print(f"Clustering metrics saved for k={k} in {metrics_file}")

        # 保存当前 k 的融合特征与聚类标签
        fusion_df_k = pd.DataFrame(fusion_features, index=sample_ids)
        fusion_df_k["Cluster"] = cluster_labels

        # 单独亚型热图（保留横坐标但只显示部分标签）
        subtype_dir = os.path.join(k_metric_dir, "subtype_heatmaps")
        plot_subtype_heatmaps(fusion_df_k, k=k, output_dir=subtype_dir)

        # 融合特征热图：隐藏底部特征标签（1到128）并显示不同聚类的样本
        fusion_sorted = fusion_df_k.sort_values("Cluster")
        data_to_plot = fusion_sorted.drop("Cluster", axis=1)
        plt.figure(figsize=(12, 8), dpi=600)
        sns.heatmap(data_to_plot, cmap="RdBu_r", cbar=True,
                    yticklabels=fusion_sorted["Cluster"], xticklabels=False)  # 隐藏特征标签，显示聚类标签
        plt.title(f"Fusion Features Heatmap (k = {k})", loc='center', fontsize=18)
        plt.xlabel("Fusion Features", fontsize=16)
        plt.ylabel("Samples", fontsize=16)
        heatmap_file = os.path.join(k_metric_dir, f"fusion_features_heatmap_k_{k}.png")
        plt.savefig(heatmap_file, dpi=600, bbox_inches='tight')
        plt.close()
        print(f"Overall Heatmap for k={k} saved to {heatmap_file}")
        plt.rcParams['font.family'] = 'Times New Roman'
        # Kaplan-Meier 生存曲线
        data_df_k = data_df.copy()
        data_df_k["Cluster"] = cluster_labels
        kmf = KaplanMeierFitter()
        plt.figure(figsize=(10, 6), dpi=1500)
        cluster_values = sorted(data_df_k["Cluster"].unique())
        colors = sns.color_palette("Set2", n_colors=len(cluster_values))
        for i, cluster in enumerate(cluster_values):
            cluster_data = data_df_k[data_df_k["Cluster"] == cluster]
            kmf.fit(durations=cluster_data["OS.time"], event_observed=cluster_data["OS"], label=f"Cluster {cluster}")
            kmf.plot_survival_function(ci_show=True, color=colors[i],linewidth=3)
        plt.title(f"Kaplan-Meier Survival Curves (k = {k})", loc='center', fontsize=40,fontweight='bold')
        plt.xlabel("Time (days)", fontsize=40,fontweight='bold')
        plt.ylabel("Survival Probability", fontsize=40,fontweight='bold')
        plt.xticks(fontsize=40, fontweight='bold')
        plt.yticks(fontsize=40, fontweight='bold')
        plt.legend(loc="best",fontsize=20)
        km_file = os.path.join(k_survival_dir, f"kaplan_meier_curves_k_{k}.pdf")
        plt.savefig(km_file, dpi=1500, bbox_inches='tight')
        plt.close()
        print(f"Kaplan-Meier curves saved for k={k} in {km_file}")

        # Log-rank 检验
        logrank_file = os.path.join(k_survival_dir, "logrank_tests.txt")
        with open(logrank_file, "w") as f:
            for c1, c2 in combinations(cluster_values, 2):
                d1 = data_df_k[data_df_k["Cluster"] == c1]
                d2 = data_df_k[data_df_k["Cluster"] == c2]
                results = logrank_test(d1["OS.time"], d2["OS.time"],
                                       event_observed_A=d1["OS"], event_observed_B=d2["OS"])
                f.write(f"Log-rank test between Cluster {c1} and Cluster {c2}: p-value = {results.p_value:.4f}\n")
        print(f"Log-rank tests saved for k={k} in {logrank_file}")

    overall_metrics_df = pd.DataFrame(overall_metrics)
    overall_metrics_df.to_csv(os.path.join(output_dir_metrics, "overall_clustering_metrics.csv"), index=False)

    # 绘制 5 个指标折线图（2 行布局）
    fig = plt.figure(figsize=(18, 10), dpi=600)
    outer_grid = gridspec.GridSpec(2, 1, height_ratios=[1, 1])

    # 第一行：3 个子图
    top_grid = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer_grid[0])
    axA = fig.add_subplot(top_grid[0])
    axB = fig.add_subplot(top_grid[1])
    axC = fig.add_subplot(top_grid[2])

    # 第二行：2 个子图均分整行
    bottom_grid = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer_grid[1])
    axD = fig.add_subplot(bottom_grid[0])
    axE = fig.add_subplot(bottom_grid[1])
    axes = [axA, axB, axC, axD, axE]

    metric_names = ["concordance", "partial_aic", "likelihood_ratio", "silhouette", "calinski_harabasz"]
    titles = ["Concordance Index", "Partial AIC", "Likelihood Ratio Test p-value",
              "Silhouette Score", "Calinski-Harabasz Index"]

    for i, metric in enumerate(metric_names):
        axes[i].plot(overall_metrics_df["k"], overall_metrics_df[metric],
                     marker='o', linewidth=2, markersize=8)
        axes[i].set_title(titles[i], fontsize=14, loc='center')
        axes[i].set_xlabel("Number of Clusters (k)", fontsize=12)
        axes[i].set_ylabel(titles[i], fontsize=12)
        axes[i].tick_params(axis='both', labelsize=10)

    plt.tight_layout()
    metrics_line_file = os.path.join(output_dir_metrics, "overall_metrics_line_plots.png")
    plt.savefig(metrics_line_file, dpi=600, bbox_inches='tight')
    plt.show()
    print(f"Overall metrics line plots saved as {metrics_line_file}")

########################################
# 7. 主流程
########################################

if __name__ == "__main__":
    # 修改为实际文件路径
    mirna_file = "mirna_standardized.tsv"
    protein_file = "protein_standardized.tsv"
    star_tpm_file = "star_tpm_standardized.tsv"
    clinical_file = "clinical.tsv"
    survival_file = "survival.tsv"
    string_file = "string_interactions.tsv"

    dataset = OmicsSurvivalDataset(mirna_file, protein_file, star_tpm_file, clinical_file, survival_file)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    input_dims = {'mirna': 1881, 'protein': 487, 'star_tpm': 6809}
    latent_dim = 128
    device = 'cpu'  # 如有 GPU, 改为 'cuda'
    n_protein_nodes = 487

    # 构建 STRING 网络
    protein_edge_index = build_protein_edge_index(string_file, protein_file, score_threshold=0.8)
    protein_edge_index = protein_edge_index.to(device)
    print("真实蛋白质网络 edge_index 形状：", protein_edge_index.shape)

    visualize_protein_network(protein_edge_index, n_protein_nodes)

    # 预训练自编码器
    autoencoder_mirna = OmicsAutoencoder(input_dims['mirna'], latent_dim)
    autoencoder_star_tpm = OmicsAutoencoder(input_dims['star_tpm'], latent_dim)
    pretrain_autoencoder(autoencoder_mirna, 'mirna', dataloader, device, num_epochs=20, lr=1e-3)
    pretrain_autoencoder(autoencoder_star_tpm, 'star_tpm', dataloader, device, num_epochs=20, lr=1e-3)

    # 构建多模态模型并复制预训练参数
    model = MultiModalSurvivalModel_GNN(input_dims, latent_dim, n_protein_nodes)
    model.mirna_encoder[0].weight.data = autoencoder_mirna.encoder[0].weight.data.clone()
    model.mirna_encoder[0].bias.data = autoencoder_mirna.encoder[0].bias.data.clone()
    model.mirna_encoder[2].weight.data = autoencoder_mirna.encoder[2].weight.data.clone()
    model.mirna_encoder[2].bias.data = autoencoder_mirna.encoder[2].bias.data.clone()
    model.star_tpm_encoder[0].weight.data = autoencoder_star_tpm.encoder[0].weight.data.clone()
    model.star_tpm_encoder[0].bias.data = autoencoder_star_tpm.encoder[0].bias.data.clone()
    model.star_tpm_encoder[2].weight.data = autoencoder_star_tpm.encoder[2].weight.data.clone()
    model.star_tpm_encoder[2].bias.data = autoencoder_star_tpm.encoder[2].bias.data.clone()

    optimizer_joint = optim.Adam(model.parameters(), lr=1e-3)
    joint_losses = train_joint(model, dataloader, optimizer_joint, num_epochs=20, device=device,
                               lambda_attn=0.1, protein_edge_index=protein_edge_index)
    torch.save(model.state_dict(), "model_joint.pth")

    plt.figure(figsize=(8, 6), dpi=600)
    plt.plot(range(1, len(joint_losses)+1), joint_losses, marker='o', color=contrast_palette[0])
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Average Cox Loss", fontsize=12)
    plt.title("Joint Training Loss Curve", loc='center', fontsize=14)
    plt.savefig("joint_training_loss.png", dpi=600, bbox_inches='tight')
    plt.show()

    # 提取融合特征
    model.eval()
    fusion_features = []
    all_attn = []
    loader_no_shuffle = DataLoader(dataset, batch_size=16, shuffle=False)
    with torch.no_grad():
        for batch in loader_no_shuffle:
            inputs = {
                'mirna': batch['mirna'].to(device),
                'star_tpm': batch['star_tpm'].to(device),
                'protein': batch['protein'].to(device)
            }
            _, fused_feature, attn_weights = model(inputs, protein_edge_index)
            fusion_features.append(fused_feature.cpu().numpy())
            all_attn.append(attn_weights.cpu().numpy())
    fusion_features = np.concatenate(fusion_features, axis=0)
    fusion_features_df = pd.DataFrame(fusion_features, index=dataset.sample_ids)
    kmeans = KMeans(n_clusters=3, random_state=42)
    cluster_labels = kmeans.fit_predict(fusion_features)
    fusion_features_df["Cluster"] = cluster_labels
    fusion_features_df.to_csv("fusion_features.csv")
    print("Fusion features saved to fusion_features.csv")

    # 平均融合特征热图，横坐标只显示部分标签
    avg_features = fusion_features_df.drop("Cluster", axis=1).groupby(fusion_features_df["Cluster"]).mean()
    plt.figure(figsize=(8, 6), dpi=600)
    sns.heatmap(avg_features, cmap="RdBu_r", cbar=True,
                annot=False, xticklabels=False, yticklabels=True)
    plt.title("Average Fusion Features per Cluster", loc='center', fontsize=14)
    plt.xlabel("Fusion Features", fontsize=12)
    plt.ylabel("Cluster", fontsize=12)
    plt.savefig("average_fusion_features_heatmap.png", dpi=600, bbox_inches='tight')
    plt.show()

    # 构造 data_df：合并 clinical 与 survival 数据，仅保留 dataset.sample_ids
    clinical_df = pd.read_csv("clinical.tsv", sep='\t', index_col=0)
    survival_df = pd.read_csv("survival.tsv", sep='\t', index_col=0)
    clinical_df = clinical_df.sort_index()
    survival_df = survival_df.sort_index()
    data_df = pd.merge(clinical_df, survival_df, left_index=True, right_index=True, how="inner")
    data_df = data_df.loc[dataset.sample_ids].copy()
    data_df.reset_index(inplace=True)
    data_df.rename(columns={'index': 'sample'}, inplace=True)

    save_clustering_and_survival_results(fusion_features, data_df, dataset.sample_ids,
                                         cluster_numbers=list(range(2, 11)))

    # SHAP 解释
    def survival_head_wrapper(x):
        x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
        with torch.no_grad():
            out = model.survival_head(x_tensor)
        return out.cpu().numpy()


    fused_tensor = torch.tensor(fusion_features, dtype=torch.float32).to(device)

    # 用于SHAP解释的masker
    masker = shap.maskers.Independent(fused_tensor.cpu().numpy())
    explainer = shap.Explainer(survival_head_wrapper, masker)
    shap_values = explainer(fused_tensor.cpu().numpy())

    # 绘制SHAP Summary Plot
    plt.figure(dpi=600)
    shap.summary_plot(shap_values, fused_tensor.cpu().numpy(),
                      feature_names=[f"F{i}" for i in range(fused_tensor.shape[1])],
                      show=False)
    plt.title("SHAP Summary Plot", loc='center', fontsize=14)
    plt.savefig("shap_summary.png", dpi=600, bbox_inches='tight')
    plt.close()
    print("SHAP summary plot saved as shap_summary.png")

    # 修改后的注意力权重图
    all_attn = np.concatenate(all_attn, axis=0)
    avg_attn = np.mean(all_attn, axis=0).squeeze()

    # 创建图表
    plt.figure(figsize=(6, 4), dpi=600)

    # 使用自定义的颜色调色板
    colors = sns.color_palette("Set2", n_colors=len(avg_attn))

    # 绘制条形图，并替换数字索引为具体的模态标签
    ax = sns.barplot(x=np.arange(len(avg_attn)), y=avg_attn, hue=np.arange(len(avg_attn)), palette=colors)

    # 去除图例
    ax.legend_.remove()

    # 设置x轴标签为 "miRNA", "Protein", "mRNA"
    modality_labels = ['miRNA', 'Protein', 'mRNA']
    ax.set_xticks(np.arange(len(avg_attn)))  # 设置x轴位置
    ax.set_xticklabels(modality_labels[:len(avg_attn)])  # 设置对应的标签

    # 设置标签
    plt.xlabel("Modality", fontsize=12)
    plt.ylabel("Average Attention Weight", fontsize=12)
    plt.title("Average Attention Weight per Modality", loc='center', fontsize=14)

    # 保存图像
    plt.savefig("attention_weights.png", dpi=600, bbox_inches='tight')
    plt.close()
    print("Attention weights plot saved as attention_weights.png")
    print(avg_attn)
