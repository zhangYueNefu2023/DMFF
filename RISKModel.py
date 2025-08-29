#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DMFF 严格评估（60/20/20） + GNN（三编码器 + 注意力融合）
+ 预训练仅用 train（防泄露）
+ 临床特征强化（低频折叠、OHE、TNM/Stage 数值化、age 样条、deep_risk 分位/交互、癌种内标准化）
+ 三模型线：Cox[strata] / Coxnet / （可选）GBSA, RSF
+ 三种融合：global weight、best-of-two、per-stratum
+ 全量指标：C-index / td-AUC / IBS
"""

# ========================= 用户设置 =========================
MIRNA_FILE    = "/Users/yuezhang/Desktop/DMFF/modelV4.0/mirna_standardized.tsv"
PROTEIN_FILE  = "/Users/yuezhang/Desktop/DMFF/modelV4.0/protein_standardized.tsv"
MRNA_FILE     = "/Users/yuezhang/Desktop/DMFF/modelV4.0/star_tpm_standardized.tsv"
CLINICAL_FILE = "/Users/yuezhang/Desktop/DMFF/modelV4.0/clinical.tsv"
SURVIVAL_FILE = "/Users/yuezhang/Desktop/DMFF/modelV4.0/survival.tsv"
STRING_FILE   = "/Users/yuezhang/Desktop/DMFF/modelV4.0/string_interactions.tsv"
RESULT_DIR    = "/Users/yuezhang/Desktop/DMFF/modelV4.0/result_strict_DMFF_GNN_plus"

SEED = 42
DEVICE = "cpu"  # 或 "cuda"

# 容量 & 正则（可按需微调）
LATENT_DIM = 192
DROPOUT = 0.2
WEIGHT_DECAY = 1e-5
MASK_AE = 0.05
MASK_TRAIN = 0.05
GAUSS_JITTER_STD = 0.00

# 训练
BATCH_SIZE = 32
EPOCHS_PRETRAIN = 15
EPOCHS_TRAIN = 100
EARLY_STOP_PATIENCE = 15
LR = 1e-3

# 融合增强
MODALITY_DROPOUT_P = 0.0
ATTN_TAU = 1.2
K_LIST = [3]
FIG_DPI = 600

# GNN/图设置
STRING_THRESH = 0.60
DROP_EDGE_P = 0.05
GNN_HIDDEN = 128
NUM_GNN_LAYERS = 2

# 其余训练细节
USE_WEIGHTED_SAMPLER = False
EVENT_POS_WEIGHT = 1.0
EMA_DECAY = 0.999
TOPK = 3
MC_DROPOUT_T = 30      # 用 MC=30 的推断做 DeepRisk(test)（可换 0）

# Cox/Coxnet 搜索
COX_GRID_PEN = [1e-3, 1e-2, 1e-1]
COX_GRID_L1  = [0.0, 0.3, 0.6]
COX_STRATA_PEN = 1e-1   # strata 版稳定的缺省惩罚

COXNET_L1RATIOS = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
COXNET_N_ALPHAS  = 60
COXNET_LOCAL_N   = 8    # 围绕最优 alpha 做局部细搜

import os, json, random, math, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.utils import concordance_index as _cindex
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.model_selection import StratifiedShuffleSplit
from copy import deepcopy
from collections import OrderedDict

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# 可选工具
try:
    from sksurv.metrics import cumulative_dynamic_auc, integrated_brier_score
    from sksurv.linear_model import CoxnetSurvivalAnalysis
    from sksurv.ensemble import RandomSurvivalForest
    from sksurv.ensemble import GradientBoostingSurvivalAnalysis
    SKSURV_OK = True
except Exception:
    print("[WARN] scikit-survival 不可用，跳过 Coxnet/RSF/GBSA/td-AUC/IBS。")
    SKSURV_OK = False

try:
    from patsy import dmatrix
    PATSY_OK = True
except Exception:
    print("[WARN] patsy 不可用，跳过年龄样条。")
    PATSY_OK = False

# PyG
try:
    from torch_geometric.nn import GCNConv, global_mean_pool
    from torch_geometric.utils import dropout_edge
    PYG_OK = True
except Exception as e:
    print("[ERROR] 需要安装 torch_geometric。", e)
    PYG_OK = False

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
plt.rcParams['font.family'] = 'Times New Roman'
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

# ---------------- 数据集 ----------------
class OmicsSurvivalDataset(Dataset):
    def __init__(self, mirna_file, protein_file, mrna_file, clinical_file, survival_file):
        self.df_mi = pd.read_csv(mirna_file, sep='\t', index_col=0).T
        self.df_pr = pd.read_csv(protein_file, sep='\t', index_col=0).T
        self.df_mr = pd.read_csv(mrna_file, sep='\t', index_col=0).T
        df_cli = pd.read_csv(clinical_file, sep='\t', index_col=0)
        df_sur = pd.read_csv(survival_file, sep='\t', index_col=0)
        for df in [self.df_mi, self.df_pr, self.df_mr, df_cli, df_sur]:
            df.index = df.index.astype(str).str.strip()
        self.df_surv = pd.merge(df_cli, df_sur, left_index=True, right_index=True, how="inner")
        common = sorted(list(set(self.df_mi.index) & set(self.df_pr.index) & set(self.df_mr.index) & set(self.df_surv.index)))
        self.df_mi = self.df_mi.loc[common]
        self.df_pr = self.df_pr.loc[common]
        self.df_mr = self.df_mr.loc[common]
        self.df_surv = self.df_surv.loc[common]
        self.sample_ids = common
        self.dim_mi = self.df_mi.shape[1]; self.dim_pr = self.df_pr.shape[1]; self.dim_mr = self.df_mr.shape[1]
    def __len__(self):
        return len(self.sample_ids)
    def __getitem__(self, i):
        sid = self.sample_ids[i]
        return {
            'mirna': torch.tensor(self.df_mi.loc[sid].values, dtype=torch.float32),
            'protein': torch.tensor(self.df_pr.loc[sid].values, dtype=torch.float32),
            'mrna': torch.tensor(self.df_mr.loc[sid].values, dtype=torch.float32),
            'duration': torch.tensor(self.df_surv.loc[sid, 'OS.time'], dtype=torch.float32),
            'event': torch.tensor(self.df_surv.loc[sid, 'OS'], dtype=torch.float32)
        }

# -------------- 图构建 --------------
def build_edge_index(string_file, protein_file, score_threshold=0.6):
    df_protein = pd.read_csv(protein_file, sep='\t', index_col=0).T
    protein_names = list(df_protein.columns)
    prot2idx = {p:i for i,p in enumerate(protein_names)}
    df = pd.read_csv(string_file, sep='\t')
    df = df[df['combined_score'] >= score_threshold].copy()
    if len(df) == 0:
        print("[WARN] STRING 过滤后无边，返回空图。")
        return torch.tensor([[],[]], dtype=torch.long), None, len(protein_names)
    df['idx1'] = df['node1'].map(prot2idx); df['idx2'] = df['node2'].map(prot2idx)
    df = df.dropna(subset=['idx1','idx2'])
    idx = df[['idx1','idx2']].values.astype(int)
    idx_sorted = np.sort(idx, axis=1)
    idx_unique = np.unique(idx_sorted, axis=0)
    edge_index = torch.tensor(idx_unique.T, dtype=torch.long)
    # 边权归一化
    from collections import defaultdict
    m = defaultdict(float)
    for _,r in df.iterrows():
        a,b = int(min(r['idx1'], r['idx2'])), int(max(r['idx1'], r['idx2']))
        m[(a,b)] = max(m[(a,b)], float(r['combined_score']))
    weights = torch.tensor([m[(a,b)] for a,b in idx_unique], dtype=torch.float32)
    weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-8)
    return edge_index, weights, len(protein_names)

# -------------- 模型 --------------
class EncoderMLP(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 512), nn.LeakyReLU(), nn.Linear(512, latent_dim))
    def forward(self, x): return self.net(x)

class OmicsAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 512), nn.LeakyReLU(), nn.Linear(512, latent_dim))
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 512), nn.LeakyReLU(), nn.Linear(512, input_dim))
    def forward(self, x): z = self.encoder(x); rec = self.decoder(z); return rec, z

class ProteinBranchGNN(nn.Module):
    def __init__(self, n_nodes, latent_dim, num_layers=2, hidden=128, dropedge_p=0.1):
        super().__init__()
        self.n_nodes = n_nodes
        self.dropedge_p = dropedge_p
        self.embed = nn.Sequential(nn.Linear(1, 64), nn.ReLU(), nn.Linear(64, latent_dim))
        self.convs = nn.ModuleList([GCNConv(latent_dim if i==0 else hidden, hidden, add_self_loops=False) for i in range(num_layers)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden) for _ in range(num_layers)])
        self.act = nn.ReLU()
        self.proj = nn.Linear(hidden, latent_dim)

    def _expand_edges(self, edge_index, edge_weight, B, device):
        N = self.n_nodes
        ei = edge_index.to(device).unsqueeze(0).repeat(B,1,1)
        offsets = (torch.arange(B, device=device) * N).view(B,1,1)
        ei = (ei + offsets).permute(1,0,2).reshape(2,-1)
        ew = edge_weight.to(device).repeat(B) if edge_weight is not None else None
        total_nodes = B * N
        # 自环
        loops = torch.arange(total_nodes, device=device)
        loops = torch.stack([loops, loops], dim=0)
        ei = torch.cat([ei, loops], dim=1)
        if ew is not None:
            ew = torch.cat([ew, torch.ones(total_nodes, device=device)], dim=0)
        # DropEdge
        if self.training and self.dropedge_p > 0:
            try:
                out = dropout_edge(ei, p=self.dropedge_p, training=True)
                if isinstance(out, tuple) and len(out) == 2:
                    ei, mask = out
                    if ew is not None: ew = ew[mask]
            except TypeError:
                pass
        return ei, ew, total_nodes

    def forward(self, x, edge_index, edge_weight=None):
        B = x.size(0); N = self.n_nodes; device = x.device
        h = self.embed(x.view(-1, 1))
        ei, ew, _ = self._expand_edges(edge_index, edge_weight, B, device)
        for conv, bn in zip(self.convs, self.bns):
            h_in = h
            h = conv(h, ei, edge_weight=ew)
            h = bn(h); h = self.act(h)
            if h.shape[-1] == h_in.shape[-1]:
                h = h + h_in
        h = self.proj(h)
        batch_vec = torch.arange(B, device=device).repeat_interleave(N)
        z = global_mean_pool(h, batch_vec)
        return z

class FusionLayer(nn.Module):
    def __init__(self, latent_dim, tau=1.0):
        super().__init__()
        self.attn = nn.Sequential(nn.Linear(latent_dim, 128), nn.LeakyReLU(), nn.Linear(128, 1))
        self.tau = tau
    def forward(self, features_list):
        attn = torch.stack([self.attn(f) for f in features_list], dim=1)
        attn = attn / max(self.tau, 1e-6)
        attn = torch.softmax(attn, dim=1)
        feats = torch.stack(features_list, dim=1)
        fused = torch.sum(attn * feats, dim=1)
        return fused, attn

class SurvivalHead(nn.Module):
    def __init__(self, latent_dim, dropout=DROPOUT):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)
        self.relu = nn.ReLU(); self.dropout = nn.Dropout(p=dropout)
        self.bn1 = nn.BatchNorm1d(512); self.bn2 = nn.BatchNorm1d(256); self.bn3 = nn.BatchNorm1d(128)
    def forward(self, x):
        x = self.dropout(self.relu(self.bn1(self.fc1(x))))
        x = self.dropout(self.relu(self.bn2(self.fc2(x))))
        x = self.dropout(self.relu(self.bn3(self.fc3(x))))
        return self.fc4(x)

class TriModalSurvival_GNN(nn.Module):
    def __init__(self, dims, latent_dim, n_nodes, tau=1.0, modality_drop_p=0.0):
        super().__init__()
        self.enc_mi = EncoderMLP(dims['mirna'], latent_dim)
        self.enc_mr = EncoderMLP(dims['mrna'],  latent_dim)
        self.gnn    = ProteinBranchGNN(n_nodes=n_nodes, latent_dim=latent_dim, num_layers=NUM_GNN_LAYERS, hidden=GNN_HIDDEN, dropedge_p=DROP_EDGE_P)
        self.fusion = FusionLayer(latent_dim, tau=tau)
        self.head   = SurvivalHead(latent_dim)
        self.modality_drop_p = modality_drop_p
    def _apply_modality_dropout(self, xs):
        if not self.training or self.modality_drop_p <= 0: return xs
        B = next(iter(xs.values())).size(0)
        keep = {k: (torch.rand(B, device=xs[k].device) > self.modality_drop_p).float().unsqueeze(1) for k in xs}
        all_drop = (1 - torch.stack(list(keep.values()), dim=0)).prod(dim=0).squeeze(1)
        for k in keep: keep[k][all_drop==1] = 1.0
        return {k: xs[k] * keep[k] for k in xs}
    def forward(self, inputs, edge_index, edge_weight=None):
        inputs = self._apply_modality_dropout(inputs)
        f_mi = self.enc_mi(inputs['mirna'])
        f_mr = self.enc_mr(inputs['mrna'])
        f_pr = self.gnn(inputs['protein'], edge_index, edge_weight)
        fused, attn = self.fusion([f_mi, f_pr, f_mr])
        r = self.head(fused)
        return r, fused, attn

# -------------- 训练 & 工具 --------------
def cox_loss(risk, durations, events):
    risk = risk.squeeze().clamp(min=-10, max=10)
    order = torch.argsort(durations, descending=True)
    r = risk[order]; e = events[order]
    exp_r = torch.exp(r)
    cum_sum = torch.cumsum(exp_r, dim=0)
    loglik = r - torch.log(cum_sum + 1e-8)
    return -torch.sum(loglik * e) / (torch.sum(e) + 1e-8)

def apply_mask(x, p):
    if p <= 0: return x
    return x * (torch.rand_like(x) > p).float()

def add_jitter(x, std=GAUSS_JITTER_STD):
    if std and std > 0: return x + torch.randn_like(x) * std
    return x

@torch.no_grad()
def cindex_on_loader(model, loader, device, edge_index, edge_weight):
    model.eval(); all_r=[]; all_t=[]; all_e=[]
    for batch in loader:
        inputs = {k: batch[k].to(device) for k in ['mirna','protein','mrna']}
        r,_,_ = model(inputs, edge_index, edge_weight)
        all_r.append(r.squeeze().cpu().numpy())
        all_t.append(batch['duration'].cpu().numpy())
        all_e.append(batch['event'].cpu().numpy())
    r = np.concatenate(all_r); t = np.concatenate(all_t); e = np.concatenate(all_e)
    return float(_cindex(t, -r, e))

def pretrain_autoencoder(ae, key, loader, device, epochs=EPOCHS_PRETRAIN, lr=LR):
    ae.to(device)
    opt = optim.Adam(ae.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    for ep in range(1, epochs+1):
        ae.train(); tot=0.0; N=0
        for batch in loader:
            x = batch[key].to(device)
            x_noisy = apply_mask(x, MASK_AE)
            rec,_ = ae(x_noisy)
            loss = nn.MSELoss()(rec, x)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()*x.size(0); N += x.size(0)
        print(f"[AE-{key}] {ep}/{epochs} Recon={tot/max(N,1):.4f}")

class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.ema = deepcopy(model).eval()
        for p in self.ema.parameters(): p.requires_grad_(False)
        self.decay = decay
    @torch.no_grad()
    def update(self, model):
        d = self.decay
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            v.copy_(v * d + msd[k] * (1. - d))
    @torch.no_grad()
    def copy_to(self, model):
        model.load_state_dict(self.ema.state_dict(), strict=True)

def average_checkpoints(state_dicts):
    avg = OrderedDict()
    for sd in state_dicts:
        for k, v in sd.items():
            if k not in avg: avg[k] = v.clone().float()
            else: avg[k] += v.float()
    for k in avg: avg[k] /= len(state_dicts)
    return avg

def train_joint(model, train_loader, val_loader, device, epochs=EPOCHS_TRAIN, lr=LR, edge_index=None, edge_weight=None):
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=5, min_lr=1e-5, verbose=True)
    ema = ModelEMA(model, decay=EMA_DECAY)

    best_val = -np.inf; best_state=None; wait=0; hist=[]
    topk = []

    for ep in range(1, epochs+1):
        # DropEdge / tau 退火
        if hasattr(model, 'gnn'):
            ramp = min(ep / 20.0, 1.0)
            model.gnn.dropedge_p = float(DROP_EDGE_P * ramp)
        if hasattr(model, 'fusion'):
            tau0, tauT = 1.0, ATTN_TAU
            ramp_tau = min(ep / 30.0, 1.0)
            model.fusion.tau = tau0 + (tauT - tau0) * ramp_tau

        model.train(); tot=0.0; N=0
        for batch in train_loader:
            inputs = {k: add_jitter(apply_mask(batch[k].to(device), MASK_TRAIN)) for k in ['mirna','protein','mrna']}
            d = batch['duration'].to(device); e = batch['event'].to(device)
            opt.zero_grad(); r,_,_ = model(inputs, edge_index, edge_weight)
            loss = cox_loss(r, d, e)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            ema.update(model)
            tot += loss.item()*d.size(0); N += d.size(0)
        avg = tot/max(N,1); hist.append(avg)

        _bak = deepcopy(model.state_dict())
        ema.copy_to(model)
        val_c = cindex_on_loader(model, val_loader, device, edge_index, edge_weight)
        model.load_state_dict(_bak, strict=True)

        scheduler.step(val_c)
        print(f"[Train] {ep}/{epochs} Loss={avg:.4f} | Val C(EMA)={val_c:.4f} | DropEdge={getattr(model.gnn,'dropedge_p',0):.3f} | tau={getattr(model.fusion,'tau',ATTN_TAU):.2f}")

        if val_c > best_val + 1e-4:
            best_val = val_c
            best_state = {k:v.cpu().clone() for k,v in ema.ema.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= EARLY_STOP_PATIENCE:
                print(f"Early stop @ {ep}. Best Val C={best_val:.4f}")
                break

        topk.append({k:v.cpu().clone() for k,v in ema.ema.state_dict().items()})
        if len(topk) > TOPK:
            topk.pop(0)

    if len(topk) > 0:
        avg_sd = average_checkpoints(topk)
        model.load_state_dict(avg_sd, strict=True)
    elif best_state is not None:
        model.load_state_dict(best_state, strict=True)

    return hist, best_val

@torch.no_grad()
def predict_all(model, loader, device, edge_index, edge_weight, mc_dropout_T=0):
    model.eval()
    if mc_dropout_T and mc_dropout_T > 0:
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.train()
            elif isinstance(m, nn.BatchNorm1d):
                m.eval()
        outs = []
        for _ in range(mc_dropout_T):
            risks=[]
            for batch in loader:
                inputs = {k: batch[k].to(device) for k in ['mirna','protein','mrna']}
                r,_,_ = model(inputs, edge_index, edge_weight)
                risks.append(r.squeeze().cpu().numpy())
            outs.append(np.concatenate(risks))
        risk_mean = np.mean(np.vstack(outs), axis=0)
        model.eval()
        return risk_mean, None, None

    risks=[]; fused=[]; attns=[]
    for batch in loader:
        inputs = {k: batch[k].to(device) for k in ['mirna','protein','mrna']}
        r,z,a = model(inputs, edge_index, edge_weight)
        risks.append(r.squeeze().cpu().numpy()); fused.append(z.cpu().numpy()); attns.append(a.cpu().numpy())
    return np.concatenate(risks), np.concatenate(fused), np.concatenate(attns)

# --------- 临床预处理 & 构造 ----------
def fold_low_freq_categories(df, cat_cols, min_count=10):
    df = df.copy()
    for c in cat_cols:
        if c not in df.columns:
            df[c] = "UNK"
        s = df[c].astype(str).fillna("UNK").replace({"nan": "UNK"})
        vc = s.value_counts(dropna=False)
        rare = set(vc[vc < min_count].index.tolist())
        df[c] = s.apply(lambda x: x if x not in rare else "UNK")
    return df

def map_stage_tnm(df):
    """粗规则把 Stage/T/N/M 变为有序数值（缺失/异常 -> 0）"""
    df = df.copy()
    stage_map = {'Stage I':1,'I':1,'Stage II':2,'II':2,'Stage III':3,'III':3,'Stage IV':4,'IV':4}
    def to_stage_num(x):
        x = str(x).upper().replace('STAGE','').strip()
        for k,v in [('IV',4),('III',3),('II',2),('I',1)]:
            if k in x: return v
        try:
            return float(x)
        except:
            return 0.0
    def tnm_num(x, letters=('T','N','M')):
        s = str(x).upper()
        for L in letters:
            if s.startswith(L):
                s = s[1:]
                break
        s = s.replace('X','0').replace('IS','0')
        s = ''.join([ch for ch in s if ch.isdigit() or ch=='.'])
        try:
            return float(s) if s!='' else 0.0
        except:
            return 0.0
    for col in ['stage','T','N','M']:
        if col not in df.columns: df[col] = "UNK"
    df['stage_num'] = df['stage'].apply(to_stage_num).astype(float)
    df['T_num'] = df['T'].apply(lambda x: tnm_num(x, ('T',))).astype(float)
    df['N_num'] = df['N'].apply(lambda x: tnm_num(x, ('N',))).astype(float)
    df['M_num'] = df['M'].apply(lambda x: tnm_num(x, ('M',))).astype(float)
    return df

class ClinicalPreprocessor:
    def __init__(self, numeric_cols, categorical_cols, min_cat_count=10):
        self.num_cols = list(numeric_cols)
        self.cat_cols = list(categorical_cols)
        self.min_cat_count = int(min_cat_count)
        self.num_medians = {}
        self.cat_uniques = {}
        self.cat_ref = {}

    def fit(self, df_train):
        df_train = df_train.copy()
        # 低频折叠
        df_train = fold_low_freq_categories(df_train, self.cat_cols, min_count=self.min_cat_count)
        # 数值：中位数
        for c in self.num_cols:
            v = pd.to_numeric(df_train.get(c, pd.Series([np.nan]*len(df_train), index=df_train.index)), errors='coerce')
            med = np.nanmedian(v.values)
            self.num_medians[c] = float(med if np.isfinite(med) else 0.0)
        # 类别：全集 + 参考（UNK 优先）
        for c in self.cat_cols:
            s = df_train.get(c, pd.Series(["UNK"]*len(df_train), index=df_train.index)).astype(str).fillna("UNK").replace({"nan":"UNK"})
            uniq = sorted(list(pd.unique(s)))
            if "UNK" not in uniq: uniq.append("UNK")
            self.cat_uniques[c] = uniq
            self.cat_ref[c] = "UNK" if "UNK" in uniq else s.value_counts().idxmax()

    def transform(self, df):
        df = df.copy()
        df = fold_low_freq_categories(df, self.cat_cols, min_count=self.min_cat_count)
        frames = []
        # 数值
        for c in self.num_cols:
            v = pd.to_numeric(df.get(c, pd.Series([np.nan]*len(df), index=df.index)), errors='coerce')
            v = v.fillna(self.num_medians[c]).astype(float)
            frames.append(pd.DataFrame({c: v.values}, index=df.index))
        # 类别 drop-first
        for c in self.cat_cols:
            s = df.get(c, pd.Series(["UNK"]*len(df), index=df.index)).astype(str).fillna("UNK").replace({"nan":"UNK"})
            known = set(self.cat_uniques[c])
            s = s.apply(lambda x: x if x in known else "UNK")
            cols = [f"{c}__{v}" for v in self.cat_uniques[c] if v != self.cat_ref[c]]
            one = pd.get_dummies(s)
            tmp = pd.DataFrame(0, index=df.index, columns=cols)
            for v in one.columns:
                if v == self.cat_ref[c]: continue
                name = f"{c}__{v}"
                if name in cols:
                    tmp[name] = one[v].values
            frames.append(tmp)
        out = pd.concat(frames, axis=1)
        return out

def add_age_spline(dfX, age_series, df=4):
    if not PATSY_OK:
        return dfX
    age = pd.to_numeric(age_series, errors='coerce')
    med = float(np.nanmedian(age.values)) if np.isfinite(np.nanmedian(age.values)) else 0.0
    age = age.fillna(med)
    bs = dmatrix(f"bs(x, df={df}, degree=3, include_intercept=False)",
                 {"x": age.values}, return_type="dataframe")
    bs.columns = [f"age_spline_{i}" for i in range(bs.shape[1])]
    bs.index = dfX.index
    out = dfX.drop(columns=[c for c in dfX.columns if c == "age"], errors="ignore").copy()
    out = pd.concat([out, bs], axis=1)
    return out

def align_and_filter_by_train(Xtr, Xva, Xte, min_std=1e-12, min_nonzero_frac=0.005):
    cols = sorted(list(set(Xtr.columns) | set(Xva.columns) | set(Xte.columns)))
    Xtr = Xtr.reindex(columns=cols, fill_value=0)
    Xva = Xva.reindex(columns=cols, fill_value=0)
    Xte = Xte.reindex(columns=cols, fill_value=0)
    Xtr_clean = Xtr.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    std = Xtr_clean.std(axis=0)
    nz_frac = (Xtr_clean.astype(bool).sum(axis=0) / len(Xtr_clean))
    keep = (std > min_std) & (nz_frac > min_nonzero_frac)
    keep_cols = list(std.index[keep])
    def _clean(df):
        df = df.reindex(columns=keep_cols, fill_value=0)
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return df
    return _clean(Xtr), _clean(Xva), _clean(Xte), keep_cols

# ------- 评估工具 -------
def to_structured_y(t, e):
    return np.array([(bool(ev), float(tt)) for ev,tt in zip(e,t)], dtype=[('event','bool'),('time','f8')])

def standardize_scores(x, eps=1e-8):
    x = np.asarray(x).reshape(-1)
    m = np.mean(x); s = np.std(x)
    if not np.isfinite(s) or s < eps: s = eps
    return (x - m) / s

def ibs_from_scores(t_tr, e_tr, t_te, e_te, r_tr, r_te, eval_times):
    """用 train 风险拟合一元 Cox 得到 SF，再在 test 上积分 IBS"""
    if not SKSURV_OK: return np.nan
    try:
        dftr = pd.DataFrame({"OS.time": t_tr, "OS": e_tr, "risk": r_tr})
        cph_tmp = CoxPHFitter(penalizer=1e-6)
        cph_tmp.fit(dftr, duration_col="OS.time", event_col="OS", show_progress=False)
        sf_te = cph_tmp.predict_survival_function(pd.DataFrame({"risk": r_te}), times=eval_times).T.values
        return float(integrated_brier_score(to_structured_y(t_tr, e_tr),
                                            to_structured_y(t_te, e_te),
                                            sf_te, eval_times))
    except Exception as ex:
        print("[WARN] IBS 失败:", ex)
        return np.nan

def search_best_weight(r_va_a, r_va_b, t_va, e_va, step=0.02):
    best_w = 0.5; best_ci = -np.inf
    A = standardize_scores(r_va_a); B = standardize_scores(r_va_b)
    for w in np.arange(0.0, 1.0+1e-9, step):
        r = w*A + (1-w)*B
        ci = float(_cindex(t_va, -r, e_va))
        if ci > best_ci + 1e-9:
            best_ci = ci; best_w = float(w)
    return best_w, best_ci

# -------------- 主流程 --------------
if __name__ == "__main__":
    assert PYG_OK, "需要安装 torch_geometric 才能运行 GNN 版本。"
    os.makedirs(RESULT_DIR, exist_ok=True)
    ds = OmicsSurvivalDataset(MIRNA_FILE, PROTEIN_FILE, MRNA_FILE, CLINICAL_FILE, SURVIVAL_FILE)
    dims = {'mirna': ds.dim_mi, 'protein': ds.dim_pr, 'mrna': ds.dim_mr}

    # 构图
    edge_index, edge_weight, n_nodes = build_edge_index(STRING_FILE, PROTEIN_FILE, score_threshold=STRING_THRESH)
    edge_index = edge_index.to(DEVICE)
    edge_weight = None if edge_weight is None else edge_weight.to(DEVICE)
    print("edge_index shape:", tuple(edge_index.shape), "; num_nodes:", n_nodes)

    # 60/20/20 stratified split
    events_all = ds.df_surv['OS'].values.astype(int)
    idx = np.arange(len(ds))
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
    trainval_idx, test_idx = next(sss1.split(idx, events_all))
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)  # 0.25 of 0.8 = 0.2
    tr_rel, va_rel = next(sss2.split(trainval_idx, events_all[trainval_idx]))
    train_idx = trainval_idx[tr_rel]; val_idx = trainval_idx[va_rel]

    # DataLoader —— 训练集可选 WeightedRandomSampler
    if USE_WEIGHTED_SAMPLER:
        e_tr_tmp = events_all[train_idx]
        weights = np.where(e_tr_tmp==1, EVENT_POS_WEIGHT, 1.0).astype(np.float32)
        sampler = WeightedRandomSampler(weights=weights, num_samples=len(train_idx), replacement=True)
        train_loader = DataLoader(Subset(ds, train_idx), batch_size=BATCH_SIZE, sampler=sampler)
    else:
        train_loader = DataLoader(Subset(ds, train_idx), batch_size=BATCH_SIZE, shuffle=True)

    val_loader   = DataLoader(Subset(ds, val_idx),   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(Subset(ds, test_idx),  batch_size=BATCH_SIZE, shuffle=False)

    # =============== AE 预训练（仅用 Train 防泄露） ===============
    pre_loader = DataLoader(Subset(ds, train_idx), batch_size=BATCH_SIZE, shuffle=True)
    ae_mi = OmicsAutoencoder(dims['mirna'], LATENT_DIM)
    ae_mr = OmicsAutoencoder(dims['mrna'],  LATENT_DIM)
    pretrain_autoencoder(ae_mi, 'mirna', pre_loader, DEVICE)
    pretrain_autoencoder(ae_mr, 'mrna',  pre_loader, DEVICE)

    # 构建 GNN 模型
    model = TriModalSurvival_GNN(dims, LATENT_DIM, n_nodes=n_nodes, tau=ATTN_TAU, modality_drop_p=MODALITY_DROPOUT_P).to(DEVICE)
    with torch.no_grad():
        for enc, ae in [(model.enc_mi, ae_mi), (model.enc_mr, ae_mr)]:
            enc.net[0].weight.data = ae.encoder[0].weight.data.clone(); enc.net[0].bias.data = ae.encoder[0].bias.data.clone()
            enc.net[2].weight.data = ae.encoder[2].weight.data.clone(); enc.net[2].bias.data = ae.encoder[2].bias.data.clone()

    # 训练（EMA + TopK + 早停 + LR 调度）
    losses, best_val = train_joint(model, train_loader, val_loader, DEVICE, edge_index=edge_index, edge_weight=edge_weight)

    # 预测（DeepRisk）
    r_tr, z_tr, a_tr = predict_all(model, train_loader, DEVICE, edge_index, edge_weight, mc_dropout_T=0)
    r_va, z_va, a_va = predict_all(model, val_loader,   DEVICE, edge_index, edge_weight, mc_dropout_T=0)
    r_te_mc, _, _    = predict_all(model, test_loader,  DEVICE, edge_index, edge_weight, mc_dropout_T=MC_DROPOUT_T)
    r_te_single, z_te, a_te = predict_all(model, test_loader, DEVICE, edge_index, edge_weight, mc_dropout_T=0)

    # DeepRisk 指标
    df_cli_all = pd.read_csv(CLINICAL_FILE, sep='\t', index_col=0).sort_index()
    df_sur_all = pd.read_csv(SURVIVAL_FILE, sep='\t', index_col=0).sort_index()
    data_df = pd.merge(df_cli_all, df_sur_all, left_index=True, right_index=True, how="inner").loc[ds.sample_ids]
    times = data_df['OS.time'].values.astype(float); events = data_df['OS'].values.astype(int)
    t_tr, t_va, t_te = times[train_idx], times[val_idx], times[test_idx]
    e_tr, e_va, e_te = events[train_idx], events[val_idx], events[test_idx]

    c_tr = float(_cindex(t_tr, -r_tr, e_tr))
    c_va = float(_cindex(t_va, -r_va, e_va))
    c_te_mc = float(_cindex(t_te, -r_te_mc, e_te))
    c_te_single = float(_cindex(t_te, -r_te_single, e_te))
    print(f"[DeepRisk] C-index Train/Val/Test(single/MC{MC_DROPOUT_T}): {c_tr:.4f}/{c_va:.4f}/{c_te_single:.4f}/{c_te_mc:.4f}")

    eval_times = [float(np.quantile(t_tr, q)) for q in (0.25, 0.5, 0.75)]
    t_auc_str = ";".join(f"{t:.3f}" for t in eval_times)
    td_te_str = ""; ibs_te = np.nan
    if SKSURV_OK:
        try:
            aucs_te,_ = cumulative_dynamic_auc(to_structured_y(t_tr, e_tr), to_structured_y(t_te, e_te), r_te_mc, eval_times)
            td_te_str = ";".join(f"{a:.4f}" for a in aucs_te)
        except Exception as e:
            print("[WARN] td-AUC(DeepRisk) 失败:", e)
        try:
            ibs_te = ibs_from_scores(t_tr, e_tr, t_te, e_te, r_tr, r_te_mc, eval_times)
        except Exception as e:
            print("[WARN] IBS(DeepRisk) 失败:", e)

    rows = [
        {"model": "DMFF-Strict (GNN) [TRAIN-60%]", "ok": True, "c_index": c_tr,
         "time_points_auc": t_auc_str, "td_auc": "", "time_points_ibs": t_auc_str,
         "ibs": np.nan, "logrank_p": np.nan, "notes": json.dumps({"split":"60/20/20","set":"train"}, ensure_ascii=False)},
        {"model": "DMFF-Strict (GNN) [VAL-20%]", "ok": True, "c_index": c_va,
         "time_points_auc": "", "td_auc": "", "time_points_ibs": "", "ibs": np.nan,
         "logrank_p": np.nan, "notes": json.dumps({"split":"60/20/20","set":"val"}, ensure_ascii=False)},
        {"model": f"DMFF-Strict (GNN,MC{MC_DROPOUT_T}) [TEST-20%]", "ok": True, "c_index": c_te_mc,
         "time_points_auc": t_auc_str, "td_auc": td_te_str, "time_points_ibs": t_auc_str,
         "ibs": ibs_te, "logrank_p": np.nan, "notes": json.dumps({"split":"60/20/20","set":"test"}, ensure_ascii=False)},
        {"model": "DMFF-Strict (GNN,Single) [TEST-20%]", "ok": True, "c_index": c_te_single,
         "time_points_auc": t_auc_str, "td_auc": td_te_str, "time_points_ibs": t_auc_str,
         "ibs": ibs_te, "logrank_p": np.nan, "notes": json.dumps({"split":"60/20/20","set":"test_single"}, ensure_ascii=False)},
    ]

    # =================== 临床协变量 + DeepRisk：特征构造 ===================
    numeric_cols = ["age", "year_of_birth.demographic", "stage_num", "T_num", "N_num", "M_num"]
    categorical_cols = [
        "disease_type", "race", "ethnicity.demographic", "code.tissue_source_site",
        "stage", "primary_diagnosis.diagnoses", "T", "N", "M"
    ]

    df_cli_all = df_cli_all.loc[ds.sample_ids]
    df_cli_all = map_stage_tnm(df_cli_all)

    df_cli_tr = df_cli_all.iloc[train_idx]
    df_cli_va = df_cli_all.iloc[val_idx]
    df_cli_te = df_cli_all.iloc[test_idx]

    # 预处理器：仅用 Train 拟合
    pre = ClinicalPreprocessor(numeric_cols, categorical_cols, min_cat_count=10)
    pre.fit(df_cli_tr)

    X_tr = pre.transform(df_cli_tr)
    X_va = pre.transform(df_cli_va)
    X_te = pre.transform(df_cli_te)

    # 年龄样条
    X_tr = add_age_spline(X_tr, df_cli_tr.get("age", pd.Series([np.nan]*len(df_cli_tr), index=df_cli_tr.index)), df=4)
    X_va = add_age_spline(X_va, df_cli_va.get("age", pd.Series([np.nan]*len(df_cli_va), index=df_cli_va.index)), df=4)
    X_te = add_age_spline(X_te, df_cli_te.get("age", pd.Series([np.nan]*len(df_cli_te), index=df_cli_te.index)), df=4)

    # 添加 deep_risk + 癌种内 z-score
    dtype_tr = df_cli_tr.get("disease_type", pd.Series(["UNK"]*len(df_cli_tr), index=df_cli_tr.index)).astype(str).fillna("UNK")
    dtype_va = df_cli_va.get("disease_type", pd.Series(["UNK"]*len(df_cli_va), index=df_cli_va.index)).astype(str).fillna("UNK")
    dtype_te = df_cli_te.get("disease_type", pd.Series(["UNK"]*len(df_cli_te), index=df_cli_te.index)).astype(str).fillna("UNK")

    def strat_z(risk, dtype):
        z = np.zeros_like(risk, dtype=float)
        for dt in np.unique(dtype):
            mask = (dtype == dt)
            z[mask] = standardize_scores(risk[mask])
        return z

    X_tr["deep_risk"]  = r_tr
    X_va["deep_risk"]  = r_va
    X_te["deep_risk"]  = r_te_mc

    X_tr["deep_risk_gz"] = strat_z(r_tr, dtype_tr.values)
    X_va["deep_risk_gz"] = strat_z(r_va, dtype_va.values)
    X_te["deep_risk_gz"] = strat_z(r_te_mc, dtype_te.values)

    # deep_risk 分位哑变量（基于 train 的 q50/q75）
    q50 = np.quantile(X_tr["deep_risk"], 0.50)
    q75 = np.quantile(X_tr["deep_risk"], 0.75)
    for X in [X_tr, X_va, X_te]:
        X["dr_ge_q50"] = (X["deep_risk"] >= q50).astype(int)
        X["dr_ge_q75"] = (X["deep_risk"] >= q75).astype(int)

    # 轻交互：deep_risk_gz * (stage_num/T_num/N_num)
    for c in ["stage_num", "T_num", "N_num"]:
        if c in X_tr.columns:
            X_tr[f"{c}*dr_gz"] = X_tr[c].values * X_tr["deep_risk_gz"].values
            X_va[f"{c}*dr_gz"] = X_va[c].values * X_va["deep_risk_gz"].values
            X_te[f"{c}*dr_gz"] = X_te[c].values * X_te["deep_risk_gz"].values

    # 按 Train 过滤并对齐
    X_tr, X_va, X_te, kept_cols = align_and_filter_by_train(X_tr, X_va, X_te)

    # ===================== 模型 1：Cox[strata(disease_type)] =====================
    # 注意：用 strata 时去掉 disease_type 的 one-hot
    drop_cols = [c for c in X_tr.columns if c.startswith("disease_type__")]
    Xtr_cs = X_tr.drop(columns=drop_cols, errors="ignore")
    Xva_cs = X_va.drop(columns=drop_cols, errors="ignore")
    Xte_cs = X_te.drop(columns=drop_cols, errors="ignore")

    df_tr_cs = Xtr_cs.copy(); df_tr_cs["OS.time"]=t_tr; df_tr_cs["OS"]=e_tr
    df_va_cs = Xva_cs.copy(); df_va_cs["OS.time"]=t_va; df_va_cs["OS"]=e_va

    cph_s = None
    try:
        cph_s = CoxPHFitter(penalizer=COX_STRATA_PEN, l1_ratio=0.0)
        cph_s.fit(df_tr_cs, duration_col="OS.time", event_col="OS",
                  show_progress=False, strata=["disease_type"])
        r_tr_cs = cph_s.predict_partial_hazard(Xtr_cs).values.reshape(-1)
        r_va_cs = cph_s.predict_partial_hazard(Xva_cs).values.reshape(-1)
        r_te_cs = cph_s.predict_partial_hazard(Xte_cs).values.reshape(-1)
        ci_tr_cs = float(_cindex(t_tr, -r_tr_cs, e_tr))
        ci_va_cs = float(_cindex(t_va, -r_va_cs, e_va))
        ci_te_cs = float(_cindex(t_te, -r_te_cs, e_te))
    except Exception as e:
        print("[WARN] Cox[strata] 拟合失败：", e)
        r_tr_cs = r_va_cs = r_te_cs = None
        ci_tr_cs = ci_va_cs = ci_te_cs = np.nan

    td_te_cs = ""; ibs_te_cs = np.nan
    if SKSURV_OK and r_te_cs is not None:
        try:
            aucs_te,_ = cumulative_dynamic_auc(to_structured_y(t_tr, e_tr), to_structured_y(t_te, e_te),
                                               r_te_cs, eval_times)
            td_te_cs = ";".join(f"{a:.4f}" for a in aucs_te)
        except Exception as e:
            print("[WARN] td-AUC(cox_strata) 失败:", e)
        try:
            ibs_te_cs = ibs_from_scores(t_tr, e_tr, t_te, e_te, r_tr_cs, r_te_cs, eval_times)
        except Exception as e:
            print("[WARN] IBS(cox_strata) 失败:", e)

    rows += [
        {"model": "DMFF-Strict (DeepRisk + Clinical Cox[strata]) [TRAIN-60%]", "ok": True, "c_index": ci_tr_cs,
         "time_points_auc": t_auc_str, "td_auc": "", "time_points_ibs": t_auc_str, "ibs": np.nan, "logrank_p": np.nan,
         "notes": json.dumps({"split":"60/20/20","set":"train_fused_cox_strata","penalizer":COX_STRATA_PEN,"l1_ratio":0.0,
                              "strata":"disease_type"}, ensure_ascii=False)},
        {"model": "DMFF-Strict (DeepRisk + Clinical Cox[strata]) [VAL-20%]", "ok": True, "c_index": ci_va_cs,
         "time_points_auc": "", "td_auc": "", "time_points_ibs": "", "ibs": np.nan, "logrank_p": np.nan,
         "notes": json.dumps({"split":"60/20/20","set":"val_fused_cox_strata","penalizer":COX_STRATA_PEN,"l1_ratio":0.0,
                              "strata":"disease_type"}, ensure_ascii=False)},
        {"model": "DMFF-Strict (DeepRisk + Clinical Cox[strata]) [TEST-20%]", "ok": True, "c_index": ci_te_cs,
         "time_points_auc": t_auc_str, "td_auc": td_te_cs, "time_points_ibs": t_auc_str, "ibs": ibs_te_cs,
         "logrank_p": np.nan, "notes": json.dumps({"split":"60/20/20","set":"test_fused_cox_strata","penalizer":COX_STRATA_PEN,
                                                   "l1_ratio":0.0,"strata":"disease_type"}, ensure_ascii=False)},
    ]

    # ===================== 模型 2：Coxnet（sksurv） =====================
    if SKSURV_OK:
        # Coxnet 不支持 strata，因此保留 disease_type 的 OHE
        y_tr = to_structured_y(t_tr, e_tr); y_va = to_structured_y(t_va, e_va); y_te = to_structured_y(t_te, e_te)

        # 初步拟合：多 l1_ratio，n_alphas 较多；选择 Val CI 最优的 (l1_ratio, alpha)
        best_ci_cnet = -np.inf; best_l1=None; best_alpha=None; best_est=None
        for l1 in COXNET_L1RATIOS:
            est = CoxnetSurvivalAnalysis(l1_ratio=float(l1), n_alphas=COXNET_N_ALPHAS, max_iter=100000, tol=1e-7)
            try:
                est.fit(X_tr.values, y_tr)
                # 按路径逐一评估
                for a in est.alphas_:
                    pred_va = -est.predict(X_va.values, alpha=a)  # 返回是负的 log-risk? 我们统一取负号成 risk
                    ci = float(_cindex(t_va, pred_va, e_va))
                    if ci > best_ci_cnet + 1e-9:
                        best_ci_cnet = ci; best_l1 = l1; best_alpha = float(a); best_est = deepcopy(est)
            except Exception as e:
                print(f"[WARN] Coxnet(l1={l1}) 拟合失败：", e)

        # 局部细搜 alpha（对数尺度 ±一圈）
        if best_est is not None and best_alpha is not None:
            alphas_local = np.geomspace(best_alpha/4, best_alpha*4, COXNET_LOCAL_N)
            for a in alphas_local:
                try:
                    pred_va = -best_est.predict(X_va.values, alpha=float(a))
                    ci = float(_cindex(t_va, pred_va, e_va))
                    if ci > best_ci_cnet + 1e-9:
                        best_ci_cnet = ci; best_alpha = float(a)
                except Exception as e:
                    pass

            r_tr_cnet = -best_est.predict(X_tr.values, alpha=best_alpha).reshape(-1)
            r_va_cnet = -best_est.predict(X_va.values, alpha=best_alpha).reshape(-1)
            r_te_cnet = -best_est.predict(X_te.values, alpha=best_alpha).reshape(-1)

            ci_tr_cnet = float(_cindex(t_tr, r_tr_cnet, e_tr))
            ci_va_cnet = float(_cindex(t_va, r_va_cnet, e_va))
            ci_te_cnet = float(_cindex(t_te, r_te_cnet, e_te))
        else:
            r_tr_cnet = r_va_cnet = r_te_cnet = None
            ci_tr_cnet = ci_va_cnet = ci_te_cnet = np.nan

        td_te_cnet = ""; ibs_te_cnet = np.nan
        if r_te_cnet is not None:
            try:
                aucs_te,_ = cumulative_dynamic_auc(y_tr, y_te, r_te_cnet, eval_times)
                td_te_cnet = ";".join(f"{a:.4f}" for a in aucs_te)
            except Exception as e:
                print("[WARN] td-AUC(coxnet) 失败:", e)
            try:
                ibs_te_cnet = ibs_from_scores(t_tr, e_tr, t_te, e_te, r_tr_cnet, r_te_cnet, eval_times)
            except Exception as e:
                print("[WARN] IBS(coxnet) 失败:", e)

        rows += [
            {"model": "DMFF-Strict (DeepRisk + Clinical Coxnet) [TRAIN-60%]", "ok": True, "c_index": ci_tr_cnet,
             "time_points_auc": t_auc_str, "td_auc": "", "time_points_ibs": t_auc_str, "ibs": np.nan, "logrank_p": np.nan,
             "notes": json.dumps({"split":"60/20/20","set":"train_fused_coxnet"}, ensure_ascii=False)},
            {"model": "DMFF-Strict (DeepRisk + Clinical Coxnet) [VAL-20%]", "ok": True, "c_index": ci_va_cnet,
             "time_points_auc": "", "td_auc": "", "time_points_ibs": "", "ibs": np.nan, "logrank_p": np.nan,
             "notes": json.dumps({"split":"60/20/20","set":"val_fused_coxnet"}, ensure_ascii=False)},
            {"model": "DMFF-Strict (DeepRisk + Clinical Coxnet) [TEST-20%]", "ok": True, "c_index": ci_te_cnet,
             "time_points_auc": t_auc_str, "td_auc": td_te_cnet, "time_points_ibs": t_auc_str, "ibs": ibs_te_cnet,
             "logrank_p": np.nan, "notes": json.dumps({"split":"60/20/20","set":"test_fused_coxnet",
                                                       "params":{"l1_ratio":best_l1,"alpha":best_alpha}}, ensure_ascii=False)},
        ]
    else:
        r_tr_cnet = r_va_cnet = r_te_cnet = None
        ci_va_cnet = -np.inf

    # ===================== 融合 1：Val 全局加权 =====================
    if (r_va_cs is not None) and (r_va_cnet is not None):
        w_glob, ci_glob = search_best_weight(r_va_cs, r_va_cnet, t_va, e_va, step=0.02)
        r_te_ens = w_glob*standardize_scores(r_te_cs) + (1-w_glob)*standardize_scores(r_te_cnet)
        ci_te_ens = float(_cindex(t_te, -r_te_ens, e_te))

        td_te_ens = ""; ibs_te_ens = np.nan
        if SKSURV_OK:
            try:
                aucs_te,_ = cumulative_dynamic_auc(to_structured_y(t_tr, e_tr), to_structured_y(t_te, e_te),
                                                   np.asarray(r_te_ens).reshape(-1), eval_times)
                td_te_ens = ";".join(f"{a:.4f}" for a in aucs_te)
            except Exception as e:
                print("[WARN] td-AUC(ensemble) 失败:", e)
            try:
                # 近似训练组合（取 0.5/0.5）
                r_tr_ens = 0.5*standardize_scores(r_tr_cs) + 0.5*standardize_scores(r_tr_cnet)
                ibs_te_ens = ibs_from_scores(t_tr, e_tr, t_te, e_te, r_tr_ens, r_te_ens, eval_times)
            except Exception as e:
                print("[WARN] IBS(ensemble) 失败:", e)

        rows.append(
            {"model": "DMFF-Strict (DeepRisk + Clinical Ensemble) [TEST-20%]",
             "ok": True, "c_index": ci_te_ens,
             "time_points_auc": t_auc_str, "td_auc": td_te_ens, "time_points_ibs": t_auc_str, "ibs": ibs_te_ens,
             "logrank_p": np.nan,
             "notes": json.dumps({"split":"60/20/20","set":"test_fused_ensemble",
                                  "w_cox": w_glob, "w_coxnet": 1-w_glob,
                                  "val_ci_cox": ci_va_cs, "val_ci_coxnet": ci_va_cnet}, ensure_ascii=False)}
        )

    # ===================== 融合 2：Best-of-two by VAL（兜底） =====================
    if (r_va_cs is not None) and (r_va_cnet is not None):
        use_cox = ci_va_cs >= ci_va_cnet
        r_te_best = r_te_cs if use_cox else r_te_cnet
        ci_te_best = float(_cindex(t_te, -r_te_best, e_te))
        td_best = ""; ibs_best = np.nan
        if SKSURV_OK:
            try:
                aucs_best,_ = cumulative_dynamic_auc(to_structured_y(t_tr, e_tr), to_structured_y(t_te, e_te),
                                                     np.asarray(r_te_best).reshape(-1), eval_times)
                td_best = ";".join(f"{a:.4f}" for a in aucs_best)
            except Exception as e:
                print("[WARN] td-AUC(best-of-two) 失败:", e)
            try:
                r_tr_best = r_tr_cs if use_cox else r_tr_cnet
                ibs_best = ibs_from_scores(t_tr, e_tr, t_te, e_te, r_tr_best, r_te_best, eval_times)
            except Exception as e:
                print("[WARN] IBS(best-of-two) 失败:", e)
        rows.append(
            {"model": "DMFF-Strict (DeepRisk + Clinical Best-of-two by VAL) [TEST-20%]",
             "ok": True, "c_index": ci_te_best,
             "time_points_auc": t_auc_str, "td_auc": td_best, "time_points_ibs": t_auc_str, "ibs": ibs_best,
             "logrank_p": np.nan,
             "notes": json.dumps({"split":"60/20/20","set":"test_fused_best_of_two",
                                  "chosen": ("cox_strata" if use_cox else "coxnet"),
                                  "val_ci_cox": ci_va_cs, "val_ci_coxnet": ci_va_cnet}, ensure_ascii=False)}
        )

    # ===================== 融合 3：Per-stratum（按癌种） =====================
    if (r_va_cs is not None) and (r_va_cnet is not None):
        unique_dt = sorted(dtype_va.unique())
        w_by_dt = {}; choose_single = {}

        for dt in unique_dt:
            mask = (dtype_va.values == dt)
            if mask.sum() < 10:
                # 数据太少：择优单模态
                ci_a = float(_cindex(t_va[mask], -r_va_cs[mask], e_va[mask])) if mask.sum()>2 else -np.inf
                ci_b = float(_cindex(t_va[mask], -r_va_cnet[mask], e_va[mask])) if mask.sum()>2 else -np.inf
                choose_single[dt] = "cox" if ci_a >= ci_b else "coxnet"
                continue
            w_dt, _ = search_best_weight(r_va_cs[mask], r_va_cnet[mask], t_va[mask], e_va[mask], step=0.05)
            w_by_dt[dt] = w_dt

        # 生成 Test 预测
        r_te_ps = np.zeros_like(t_te, dtype=float)
        zs_te_cs   = standardize_scores(r_te_cs)   if r_te_cs is not None else None
        zs_te_cnet = standardize_scores(r_te_cnet) if r_te_cnet is not None else None
        for i in range(len(r_te_ps)):
            dt = dtype_te.iloc[i]
            if dt in choose_single:
                r_te_ps[i] = r_te_cs[i] if choose_single[dt]=="cox" else r_te_cnet[i]
            else:
                w = w_by_dt.get(dt, 0.5)
                r_te_ps[i] = w*zs_te_cs[i] + (1-w)*zs_te_cnet[i]

        ci_te_ps = float(_cindex(t_te, -r_te_ps, e_te))
        td_ps = ""; ibs_ps = np.nan
        if SKSURV_OK:
            try:
                aucs_ps,_ = cumulative_dynamic_auc(to_structured_y(t_tr, e_tr), to_structured_y(t_te, e_te),
                                                   np.asarray(r_te_ps).reshape(-1), eval_times)
                td_ps = ";".join(f"{a:.4f}" for a in aucs_ps)
            except Exception as e:
                print("[WARN] td-AUC(per-stratum) 失败:", e)
            try:
                # 近似 train 组合
                r_tr_ps = 0.5*standardize_scores(r_tr_cs) + 0.5*standardize_scores(r_tr_cnet)
                ibs_ps = ibs_from_scores(t_tr, e_tr, t_te, e_te, r_tr_ps, r_te_ps, eval_times)
            except Exception as e:
                print("[WARN] IBS(per-stratum) 失败:", e)

        rows.append(
            {"model": "DMFF-Strict (DeepRisk + Clinical Ensemble per-stratum) [TEST-20%]",
             "ok": True, "c_index": ci_te_ps,
             "time_points_auc": t_auc_str, "td_auc": td_ps, "time_points_ibs": t_auc_str, "ibs": ibs_ps,
             "logrank_p": np.nan,
             "notes": json.dumps({"split":"60/20/20","set":"test_fused_ensemble_per_stratum",
                                  "n_strata": len(unique_dt)}, ensure_ascii=False)}
        )

    # 保存 CSV
    csv_path = os.path.join(RESULT_DIR, "survival_model_results_GNN_plus.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"[Metrics CSV] -> {csv_path}")

    # TEST 聚类 + KM（可选）
    if 'z_te' in locals() and z_te is not None and len(z_te) >= max(K_LIST):
        for k in K_LIST:
            if len(z_te) < k: continue
            k_dir = os.path.join(RESULT_DIR, f"test_k_{k}"); os.makedirs(k_dir, exist_ok=True)
            km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
            labels = km.fit_predict(z_te)
            try: sil = silhouette_score(z_te, labels)
            except: sil = np.nan
            try: ch = calinski_harabasz_score(z_te, labels)
            except: ch = np.nan
            with open(os.path.join(k_dir, "clustering_metrics_test.txt"), 'w') as f:
                f.write(f"k={k}\nSilhouette={sil}\nCalinski-Harabasz={ch}\n")
            dd = pd.DataFrame({"t": data_df['OS.time'].values.astype(float)[test_idx],
                               "e": data_df['OS'].values.astype(int)[test_idx],
                               "Cluster": labels})
            kmf = KaplanMeierFitter(); plt.figure(figsize=(10,6), dpi=FIG_DPI)
            cvs = sorted(dd['Cluster'].unique()); colors = sns.color_palette("Set2", n_colors=len(cvs))
            for i,c in enumerate(cvs):
                sub = dd[dd['Cluster']==c]
                kmf.fit(durations=sub['t'], event_observed=sub['e'], label=f"Cluster {c}")
                kmf.plot_survival_function(ci_show=True, color=colors[i], linewidth=3)
            plt.title(f"KM (TEST, k={k})"); plt.xlabel("Time (days)"); plt.ylabel("Survival Probability")
            plt.savefig(os.path.join(k_dir, f"km_test_k{k}.pdf"), bbox_inches='tight'); plt.close()

    # TEST 平均注意力
    if 'a_te' in locals() and a_te is not None:
        avg_attn = np.mean(a_te, axis=0).squeeze()
        plt.figure(figsize=(6,4), dpi=FIG_DPI)
        colors = sns.color_palette("Set2", n_colors=len(avg_attn))
        ax = sns.barplot(x=np.arange(len(avg_attn)), y=avg_attn, hue=np.arange(len(avg_attn)), palette=colors)
        ax.legend_.remove()
        ax.set_xticks(np.arange(len(avg_attn))); ax.set_xticklabels(['miRNA','Protein','mRNA'][:len(avg_attn)])
        plt.xlabel("Modality"); plt.ylabel("Average Attention (TEST)")
        plt.title("Average Attention per Modality (TEST)")
        plt.savefig(os.path.join(RESULT_DIR, "attention_weights_TEST.tiff"), dpi=FIG_DPI, bbox_inches='tight'); plt.close()

    print("[DONE] 严格版训练 + 加强临床 + 多策略融合 完成，输出目录 ->", RESULT_DIR)
