"""
sota_baselines.py — SOTA Baseline Comparison for DA-MetaForecaster
===================================================================
Models: DLinear, iTransformer, N-HiTS, TimeMixer
Same data pipeline as patchtst_macro.py:
  - 12 features (8 price/vol + 4 macro)
  - Train 2000-2015 | Val 2016-2017 | Test 2018-2023
  - Horizons: 1d, 5d, 10d, 21d
  - Per-horizon seeds: H=1d→142, H=5d→542, H=10d→1042, H=21d→2142
  - MAE loss, Adam, early-stop patience=30, min 150 epochs
Reports: Overall DA, Crash DA, Boom DA — ready to paste into Table 4.
Run on Colab T4. Expected runtime: ~40-60 min total.
"""

import sys, os, math, warnings
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import importlib as _il
for _pkg, _pip in [("yfinance","yfinance"), ("tqdm","tqdm"),
                   ("sklearn","scikit-learn"), ("pandas_datareader","pandas-datareader"),
                   ("scipy","scipy")]:        # scipy required for DM test
    if _il.util.find_spec(_pkg) is None:
        import subprocess; subprocess.run(["pip","install","-q",_pip], check=True)
del _il

_IN_COLAB = "google.colab" in sys.modules or os.path.exists("/content")
if _IN_COLAB:
    try:
        from google.colab import drive as _gd
        _gd.mount("/content/drive", force_remount=False)
        OUT = "/content/drive/MyDrive/damf_output"
    except Exception:
        OUT = "/content/damf_output"
else:
    try:    OUT = os.path.dirname(os.path.abspath(__file__))
    except: OUT = "."
os.makedirs(OUT, exist_ok=True)

import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error
from tqdm.auto import tqdm
warnings.filterwarnings("ignore")

import time as _time
SCRIPT_START = _time.time()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# Determinism: must be set before any CUDA operations
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

# ── Log full environment for reproducibility ───────────────────────
import platform
ENV_INFO = {
    "python":     platform.python_version(),
    "torch":      torch.__version__,
    "numpy":      np.__version__,
    "pandas":     pd.__version__,
    "cuda":       torch.version.cuda if torch.cuda.is_available() else "cpu",
    "gpu":        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    "gpu_mem_gb": round(torch.cuda.get_device_properties(0).total_memory/1e9, 1)
                  if torch.cuda.is_available() else 0,
    "platform":   platform.platform(),
}
print("Environment:", ENV_INFO)

# ── constants (identical to patchtst_macro.py) ─────────────────────
HORIZONS   = [5, 10, 21]     # H=1d removed — daily returns too volatile for meaningful DA comparison
BASE_FEAT  = ["log_ret","vix","mom5","mom21","vol21","vol_ratio","drawdown","vix_change"]
MACRO_FEAT = ["yield_slope","yield_slope_chg","credit_spread","credit_spread_chg"]
FEATURES   = BASE_FEAT + MACRO_FEAT
N_FEAT     = len(FEATURES)   # 12
LOOK_BACK  = 42
TRAIN_END  = "2015-12-31"; VAL_END = "2017-12-31"
EPOCHS = 500; PATIENCE = 30; MIN_EPOCHS = 150; BS = 32; LR = 1e-3
REG_NAMES = {0:"Normal", 1:"Boom", 2:"Recovery", 3:"Crash"}

# ═══════════════════════════════════════════════════════════════════
# SECTION 1 | DATA  (same as patchtst_macro.py)
# ═══════════════════════════════════════════════════════════════════
print("\n--- DATA LOADING ---")
import yfinance as yf
import pandas_datareader.data as web

# ── Data cache: saves processed df on first run, reloads on subsequent runs ──
# This guarantees identical data across machines/dates regardless of API revisions.
_CACHE = os.path.join(OUT, "sota_data_cache.csv")
_LOAD_CACHE = os.path.exists(_CACHE)

if _LOAD_CACHE:
    df = pd.read_csv(_CACHE, parse_dates=["date"])
    print(f"Loaded cached data: {len(df)} rows from {_CACHE}")
else:
    sp  = yf.download("^GSPC", start="2000-01-01", end="2023-12-31",
                      auto_adjust=True, progress=False)["Close"].squeeze()
    vix = yf.download("^VIX",  start="2000-01-01", end="2023-12-31",
                      auto_adjust=True, progress=False)["Close"].squeeze()
    raw = pd.DataFrame({"date": pd.to_datetime(sp.index),
                        "close": sp.values,
                        "vix":   vix.reindex(sp.index).values}).dropna()

    try:
        yc_raw = web.DataReader("T10Y2Y", "fred", "1999-01-01", "2023-12-31")
        yc_raw.index = pd.to_datetime(yc_raw.index)
    except Exception:
        tnx = yf.download("^TNX", start="1999-01-01", end="2023-12-31",
                          auto_adjust=True, progress=False)["Close"].squeeze()
        irx = yf.download("^IRX", start="1999-01-01", end="2023-12-31",
                          auto_adjust=True, progress=False)["Close"].squeeze()
        yc_raw = pd.DataFrame({"T10Y2Y": (tnx-irx).values}, index=pd.to_datetime(tnx.index))

    hyg = yf.download("HYG", start="1999-01-01", end="2023-12-31",
                      auto_adjust=True, progress=False)["Close"].squeeze()
    ief = yf.download("IEF", start="1999-01-01", end="2023-12-31",
                      auto_adjust=True, progress=False)["Close"].squeeze()
    hyg.index = pd.to_datetime(hyg.index); ief.index = pd.to_datetime(ief.index)
    credit_hyg = (np.log(ief / ief.shift(63)) - np.log(hyg / hyg.shift(63))) * 100

    _vix_s  = pd.Series(yf.download("^VIX","1999-01-01","2023-12-31",
                         auto_adjust=True, progress=False)["Close"].squeeze().values,
                        index=pd.to_datetime(yf.download("^VIX","1999-01-01","2023-12-31",
                         auto_adjust=True, progress=False).index))
    _vix_z  = (_vix_s - _vix_s.rolling(252).mean()) / (_vix_s.rolling(252).std() + 1e-8)
    _overlap = credit_hyg.dropna().index
    _hyg_m, _hyg_s = credit_hyg.loc[_overlap].mean(), credit_hyg.loc[_overlap].std()
    _vix_proxy = _vix_z * _hyg_s + _hyg_m
    credit_combined = credit_hyg.copy()
    credit_combined.loc[credit_combined.isna().index] = _vix_proxy.reindex(
        credit_combined[credit_combined.isna()].index)
    hy_raw = credit_combined.to_frame("BAMLH0A0HYM2")

    df = raw.set_index("date")
    df["yield_slope"]       = yc_raw["T10Y2Y"].reindex(df.index).ffill().bfill().values
    df["yield_slope_chg"]   = df["yield_slope"].diff(1)
    df["credit_spread"]     = hy_raw["BAMLH0A0HYM2"].reindex(df.index).ffill().bfill().values
    df["credit_spread_chg"] = df["credit_spread"].diff(1)
    df = df.reset_index()

    df["log_ret"]    = np.log(df["close"]/df["close"].shift(1))
    df["mom5"]       = df["log_ret"].rolling(5).sum()
    df["mom21"]      = df["log_ret"].rolling(21).sum()
    df["vol21"]      = df["log_ret"].rolling(21).std()
    df["vix"]        = df["vix"]/100.0
    df["vol5"]       = df["log_ret"].rolling(5).std()
    df["vol_ratio"]  = df["vol5"]/(df["vol21"]+1e-8)
    df["drawdown"]   = df["close"]/df["close"].rolling(63).max()-1
    df["vix_change"] = df["vix"].pct_change(1)
    df = df.dropna().reset_index(drop=True)

    df["_dd252"]   = df["close"]/df["close"].rolling(252).max()-1
    df["_trend63"] = df["log_ret"].rolling(63).sum()
    _dist = (df["_dd252"] <= -0.10) | (df["drawdown"] <= -0.10)
    _bull = df["_trend63"] >= 0.02
    df["regime"] = np.where(~_dist & _bull, 1,
                   np.where(~_dist & ~_bull, 0,
                   np.where( _dist & _bull,  2, 3))).astype(int)
    df = df.dropna().reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])

    df.to_csv(_CACHE, index=False)
    print(f"Data cached to {_CACHE} ({len(df)} rows) — reused on future runs")

df_tr = df[df["date"] <= TRAIN_END].reset_index(drop=True)
df_va = df[(df["date"] > TRAIN_END) & (df["date"] <= VAL_END)].reset_index(drop=True)
df_te = df[df["date"] > VAL_END].reset_index(drop=True)
print(f"Train:{len(df_tr)}  Val:{len(df_va)}  Test:{len(df_te)}")

def build_windows(split, H):
    X, y_raw, y_det, trend_arr, reg_arr = [], [], [], [], []
    vals = split[FEATURES].values.astype(np.float32)
    ret  = split["log_ret"].values.astype(np.float32)
    reg  = split["regime"].values.astype(int)
    for i in range(LOOK_BACK, len(vals)-H+1):
        raw_y = float(np.sum(ret[i:i+H]))
        t_val = float(np.mean(ret[i-LOOK_BACK:i])) * H
        X.append(vals[i-LOOK_BACK:i])
        y_raw.append(raw_y); y_det.append(raw_y - t_val)
        trend_arr.append(t_val); reg_arr.append(int(reg[i-1]))
    return (np.array(X, np.float32), np.array(y_raw, np.float32),
            np.array(y_det, np.float32), np.array(trend_arr, np.float32),
            np.array(reg_arr, int))

# ═══════════════════════════════════════════════════════════════════
# SECTION 2 | MODELS
# ═══════════════════════════════════════════════════════════════════

# ── Instance Norm helper (per-sample, per-channel) ─────────────────
def inst_norm(x):
    """x: [B, L, C] → normalised [B, L, C]"""
    mu = x.mean(dim=1, keepdim=True)
    sigma = x.std(dim=1, keepdim=True) + 1e-5
    return (x - mu) / sigma


# ── 1. DLinear ─────────────────────────────────────────────────────
# "Are Transformers Effective for Time Series Forecasting?" Zeng et al., AAAI 2023
class DLinear(nn.Module):
    """Channel-independent seasonal-trend decomposition + linear projection."""
    def __init__(self, L=LOOK_BACK, C=N_FEAT, kernel=25):  # kernel=25: original paper default
        super().__init__()
        self.kernel = kernel
        # Shared linear layers applied to each channel independently
        self.seasonal_linear = nn.Linear(L, 1)
        self.trend_linear    = nn.Linear(L, 1)
        # Learned aggregation across channels
        self.channel_agg = nn.Linear(C, 1, bias=False)

    def _moving_avg(self, x):
        # x: [B, L, C] → trend [B, L, C] via symmetric moving average
        B, L, C = x.shape
        k = self.kernel
        xp = x.permute(0, 2, 1)                  # [B, C, L]
        front = xp[:, :, :1].expand(-1, -1, k//2)
        end   = xp[:, :, -1:].expand(-1, -1, k//2)
        xpad  = torch.cat([front, xp, end], dim=2)  # [B, C, L+k-1]
        w     = torch.ones(1, 1, k, device=x.device) / k
        # conv per channel
        xpad_flat = xpad.reshape(B*C, 1, L+k-1)
        trend_flat = F.conv1d(xpad_flat, w)        # [B*C, 1, L]
        return trend_flat.reshape(B, C, L).permute(0, 2, 1)  # [B, L, C]

    def forward(self, x):
        x = inst_norm(x)
        trend    = self._moving_avg(x)
        seasonal = x - trend
        # Per-channel projections: [B, C, L] → [B, C, 1]
        s = self.seasonal_linear(seasonal.permute(0, 2, 1))   # [B, C, 1]
        t = self.trend_linear(trend.permute(0, 2, 1))         # [B, C, 1]
        per_ch = (s + t).squeeze(-1)                          # [B, C]
        return self.channel_agg(per_ch).squeeze(-1)           # [B]


# ── 2. iTransformer ────────────────────────────────────────────────
# "iTransformer: Inverted Transformers Are Effective for Time Series" Liu et al., ICLR 2024
class iTransformer(nn.Module):
    """Variates as tokens; attention captures cross-variate correlations."""
    def __init__(self, L=LOOK_BACK, C=N_FEAT, D=128, n_heads=4, n_layers=3):  # 3 layers: matches PatchTST depth
        super().__init__()
        self.embed     = nn.Linear(L, D)
        self.pos_embed = nn.Parameter(torch.randn(1, C, D) * 0.02)
        # dropout=0.1 matches PatchTST (same regularisation level for fair comparison)
        # DLinear/N-HiTS/TimeMixer have no internal dropout by design (they are simpler architectures)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=D, nhead=n_heads, dim_feedforward=D*4,
            dropout=0.1, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(D)
        self.head = nn.Linear(D, 1)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        # x: [B, L, C] → transpose to treat each variate as a token
        B, L, C = x.shape
        xv = x.permute(0, 2, 1)                   # [B, C, L]
        xv = (xv - xv.mean(-1, keepdim=True)) / (xv.std(-1, keepdim=True) + 1e-5)
        emb = self.embed(xv) + self.pos_embed      # [B, C, D]
        rep = self.norm(self.encoder(emb)).mean(1)  # [B, D] — pool over variates
        return self.head(rep).squeeze(-1)           # [B]


# ── 3. N-HiTS ──────────────────────────────────────────────────────
# "N-HiTS: Neural Hierarchical Interpolation for Time Series" Challu et al., AAAI 2023
class _NHiTSBlock(nn.Module):
    def __init__(self, L, C, D, pool_size):
        super().__init__()
        self.pool_size = pool_size
        pooled_L = math.ceil(L / max(pool_size, 1))
        in_dim = pooled_L * C
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, D), nn.ReLU(),
            nn.Linear(D, D // 2), nn.ReLU(),
            nn.Linear(D // 2, 1))

    def forward(self, x):
        B, L, C = x.shape
        if self.pool_size > 1:
            # avg_pool matches original N-HiTS paper (not max_pool)
            xp = F.avg_pool1d(x.permute(0,2,1), self.pool_size,
                              ceil_mode=True).permute(0,2,1)
        else:
            xp = x
        return self.mlp(xp.reshape(B, -1))   # [B, 1]

class NHiTS(nn.Module):
    """Three-stack hierarchical architecture operating at different time resolutions."""
    def __init__(self, L=LOOK_BACK, C=N_FEAT, D=256):
        super().__init__()
        # inst_norm used (not LayerNorm) — consistent with DLinear/TimeMixer/PatchTST
        # LayerNorm([L,C]) would normalise jointly over both axes, breaking channel independence
        self.blocks = nn.ModuleList([
            _NHiTSBlock(L, C, D, pool_size=1),   # 42 steps — full resolution
            _NHiTSBlock(L, C, D, pool_size=3),   # 14 steps — weekly resolution
            _NHiTSBlock(L, C, D, pool_size=7),   # 6 steps  — bi-weekly resolution
        ])

    def forward(self, x):
        x = inst_norm(x)   # per-channel instance norm — same as all other models
        return sum(b(x) for b in self.blocks).squeeze(-1)  # [B]


# ── 4. TimeMixer ───────────────────────────────────────────────────
# "TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting" Wang et al., ICLR 2024
class TimeMixer(nn.Module):
    """Multi-scale seasonal/trend decomposition with MLP mixing."""
    def __init__(self, L=LOOK_BACK, C=N_FEAT, D=64):
        super().__init__()
        self.kernels = [5, 11, 21]    # short/medium/long MA windows
        n_scales = len(self.kernels)
        in_dim = L * C
        # Separate seasonal and trend MLPs per scale
        self.seasonal_mlp = nn.ModuleList([
            nn.Sequential(nn.Linear(in_dim, D), nn.GELU(), nn.Linear(D, D // 2))
            for _ in range(n_scales)])
        self.trend_mlp = nn.ModuleList([
            nn.Sequential(nn.Linear(in_dim, D), nn.GELU(), nn.Linear(D, D // 2))
            for _ in range(n_scales)])
        self.head = nn.Linear(D * n_scales, 1)

    def _ma(self, x, k):
        B, L, C = x.shape
        xp = x.permute(0, 2, 1)                    # [B, C, L]
        front = xp[:, :, :1].expand(-1, -1, k//2)
        end   = xp[:, :, -1:].expand(-1, -1, k//2)
        xpad  = torch.cat([front, xp, end], dim=2)
        w = torch.ones(1, 1, k, device=x.device) / k
        flat = xpad.reshape(B*C, 1, xpad.shape[-1])
        trend = F.conv1d(flat, w).reshape(B, C, L).permute(0, 2, 1)
        return trend

    def forward(self, x):
        B = x.shape[0]
        xn = inst_norm(x)
        feats = []
        for i, k in enumerate(self.kernels):
            trend    = self._ma(xn, k)
            seasonal = xn - trend
            feats.append(self.seasonal_mlp[i](seasonal.reshape(B, -1)))
            feats.append(self.trend_mlp[i](trend.reshape(B, -1)))
        combined = torch.cat(feats, dim=-1)    # [B, D * n_scales]
        return self.head(combined).squeeze(-1)  # [B]


# ── 5. LSTM ────────────────────────────────────────────────────────
# Matches v8 architecture: 2 layers, 128 hidden, all 12 features
# Re-run here on 2018-2023 test split (v8 used 2020-2024 — different period)
class LSTMBaseline(nn.Module):
    """Sequence LSTM over L=42 steps × 12 features → single scalar."""
    def __init__(self, C=N_FEAT, hidden=128, n_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(C, hidden, n_layers, batch_first=True,
                            dropout=0.1 if n_layers > 1 else 0.0)
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, 1)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        # x: [B, L, C] — no inst_norm; LSTM handles varying scales via gate saturation
        out, _ = self.lstm(x)          # [B, L, hidden]
        rep = self.norm(out[:, -1, :]) # last time step
        return self.head(rep).squeeze(-1)


# ── PatchTST (same as patchtst_macro.py Phase 1 — for DM test reference) ──
PATCH_LEN = 16; PATCH_STR = 8
N_PATCHES = (LOOK_BACK - PATCH_LEN) // PATCH_STR + 1
D_MODEL = 128; N_HEADS = 8; N_LAYERS = 3; FFN_DIM = 256

class PatchTST(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Linear(PATCH_LEN, D_MODEL)
        self.pos_embed   = nn.Parameter(torch.randn(1, N_PATCHES, D_MODEL) * 0.02)
        enc = nn.TransformerEncoderLayer(d_model=D_MODEL, nhead=N_HEADS,
              dim_feedforward=FFN_DIM, dropout=0.1, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=N_LAYERS)
        self.norm = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, 1)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        B, L, C = x.shape
        xm = x.mean(1, keepdim=True); xs = x.std(1, keepdim=True) + 1e-5
        xn = ((x - xm) / xs).permute(0, 2, 1).reshape(B*C, L)
        patches = xn.unfold(1, PATCH_LEN, PATCH_STR)
        emb = self.patch_embed(patches) + self.pos_embed
        enc = self.norm(self.encoder(emb))
        rep = enc.mean(1).reshape(B, C, D_MODEL).mean(1)
        return self.head(rep).squeeze(-1)


# ═══════════════════════════════════════════════════════════════════
# SECTION 3 | TRAINING + EVALUATION
# ═══════════════════════════════════════════════════════════════════
# PatchTST runs first — its predictions are used as reference for DM tests
MODELS = {
    "PatchTST":      PatchTST,      # Phase-1 backbone — DM test reference
    "DLinear":       DLinear,       # AAAI 2023 — standard non-transformer baseline
    "iTransformer":  iTransformer,  # ICLR 2024 — inverted attention
    "N-HiTS":        NHiTS,         # AAAI 2023 — hierarchical interpolation
    "TimeMixer":     TimeMixer,     # ICLR 2024 — multiscale mixing
}

# Per-model learning rates — overrides global LR where needed
# iTransformer and TimeMixer overfit immediately at lr=1e-3; lr=1e-4 required
MODEL_LR = {
    "PatchTST":     1e-3,
    "DLinear":      1e-3,
    "iTransformer": 1e-4,
    "N-HiTS":       1e-3,
    "TimeMixer":    1e-4,
}

CLASSICAL_MODELS = []   # LSTM/AR5/XGBoost excluded — comparison scoped to transformer-era models

# ── Hyperparameter summary (printed + saved to report) ─────────────
HPARAM_SUMMARY = f"""
HYPERPARAMETERS
===============
Environment: {ENV_INFO}

Shared training (all models):
  Loss          : MAE (F.l1_loss) — no class weights, threshold=0 (not tuned)
  Optimizer     : Adam  β1=0.9  β2=0.999  ε=1e-8  weight_decay=0.0
  Learning rate : PatchTST/DLinear/N-HiTS=1e-3  |  iTransformer/TimeMixer=1e-4 (1e-3 causes early overfitting)
  LR schedule   : CosineAnnealingLR  T_max=500  eta_min=1e-5
  Batch size    : 32  |  Grad clip norm: 1.0
  Early-stop    : patience=30  min_epochs=150  max_epochs=500
  Seeds         : H=5d→542  H=10d→1042  H=21d→2142 (per-horizon, reset per model; H=1d excluded)
  Determinism   : cudnn.deterministic=True  benchmark=False  DataLoader generator=fixed
  Runs          : 1 run per model per horizon (single-seed; no multi-run averaging)
  DA threshold  : sign(pred)==sign(actual), threshold=0.0 (fixed, not tuned on test set)

Data split (strict chronological, no shuffle):
  Train : 2000-01-01 → 2015-12-31
  Val   : 2016-01-01 → 2017-12-31
  Test  : 2018-01-01 → 2023-12-31
  Lookback window L=42 | Target: detrended H-day cumulative log return

Deep learning model architectures (α=lr=1e-3, β1=0.9, β2=0.999, ε=1e-8 shared):
  PatchTST     : L=42 P=16 S=8 N_patches=4 D=128 heads=8 layers=3 FFN=256 dropout=0.1  ~400K θ
  LSTM         : hidden=128 layers=2 dropout=0.1 final-timestep pooling  ~270K θ
  DLinear      : MA kernel=25  channel-independent linear L→1  learned channel aggregation  ~3K θ
  iTransformer : D=128 heads=4 layers=3 dropout=0.1 FFN=512  variates-as-tokens  ~270K θ
  N-HiTS       : D=256 pool_sizes=[1,3,7]→[42,14,6] steps  avg_pool (orig. paper)  ~800K θ
  TimeMixer    : D=64 MA_kernels=[5,11,21] seasonal+trend MLP per scale  ~200K θ

Classical model hyperparameters:
  AR(5)    : p=5 lags of log_ret only, OLS (LinearRegression), no regularisation  θ=5 coefs+bias
  XGBoost  : n_estimators=100  max_depth=4  learning_rate=0.1  subsample=0.8
             colsample_bytree=0.8  early_stopping_rounds=20  eval_metric=mae  random_state=seed

Normalisation: per-channel instance norm (mean/std over L) inside each deep learning model.
               AR5/XGBoost use raw feature values — no normalisation (tree/linear methods are scale-invariant).
"""
print(HPARAM_SUMMARY)

def train_model(model, Xtr, ytr_det, Xva, yva_det, seed=42, lr=LR):
    # Fixed generator ensures DataLoader shuffle is identical across runs
    g = torch.Generator(); g.manual_seed(seed)
    dl  = DataLoader(TensorDataset(torch.tensor(Xtr), torch.tensor(ytr_det)),
                     BS, shuffle=True, drop_last=False, generator=g)
    Xva_t = torch.tensor(Xva).to(DEVICE)
    yva_t = torch.tensor(yva_det).to(DEVICE)
    # Explicit Adam betas/eps — don't rely on PyTorch version defaults
    opt = torch.optim.Adam(model.parameters(), lr=lr,
                           betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=500, eta_min=lr*0.01)
    best_val, best_state, no_imp, best_ep = float("inf"), None, 0, 0
    bar = tqdm(range(1, EPOCHS+1), desc=f"  {type(model).__name__}", leave=False)
    for ep in bar:
        model.train()
        for Xb, yb in dl:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            F.l1_loss(model(Xb), yb).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sch.step()
        model.eval()
        with torch.no_grad():
            vl = F.l1_loss(model(Xva_t), yva_t).item()
        if vl < best_val:
            best_val = vl; best_ep = ep
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
        bar.set_postfix(val=f"{vl:.5f}", best=f"{best_val:.5f}", p=f"{no_imp}/{PATIENCE}")
        if no_imp >= PATIENCE and ep >= MIN_EPOCHS:
            break
    model.load_state_dict(best_state)
    print(f"    converged ep={best_ep}  best_val={best_val:.5f}")
    return model, best_ep


def dm_test(e1, e2):
    """Diebold-Mariano test (Harvey et al. correction). Returns (stat, p_value).
    e1, e2: forecast error arrays. H0: equal predictive accuracy."""
    from scipy import stats
    d = e1**2 - e2**2          # loss differential (MSE-based)
    n = len(d)
    d_bar = np.mean(d)
    # Newey-West variance (lag=1)
    gamma0 = np.var(d, ddof=1)
    gamma1 = np.cov(d[:-1], d[1:], ddof=1)[0, 1] if n > 2 else 0.0
    var_d = (gamma0 + 2*gamma1) / n
    if var_d <= 0:
        return float("nan"), float("nan")
    dm_stat = d_bar / np.sqrt(var_d)
    # Harvey et al. (1997) small-sample correction
    dm_stat *= np.sqrt((n + 1 - 2 + 1/n) / n)
    p_val = 2 * stats.t.sf(abs(dm_stat), df=n-1)
    return float(dm_stat), float(p_val)


def evaluate(model, Xte, yte_raw, yte_det, tte, rte, patchtst_pred_raw=None):
    """Returns metrics dict. If patchtst_pred_raw provided, computes DM test vs PatchTST."""
    model.eval()
    Xte_t = torch.tensor(Xte).to(DEVICE)
    with torch.no_grad():
        pred_det = model(Xte_t).cpu().numpy()
    pred_raw = pred_det + tte

    def _da(p, a, idx=None):
        if idx is not None: p, a = p[idx], a[idx]
        return float(np.mean(np.sign(p) == np.sign(a))) if len(p) >= 3 else float("nan")

    reg_da = {}
    for r, nm in REG_NAMES.items():
        idx = np.where(rte == r)[0]
        reg_da[nm] = _da(pred_raw, yte_raw, idx)

    rmse = float(np.sqrt(mean_squared_error(yte_raw, pred_raw)))

    # Diebold-Mariano vs PatchTST
    dm_stat, dm_pval = float("nan"), float("nan")
    if patchtst_pred_raw is not None:
        e_sota    = pred_raw - yte_raw
        e_patch   = patchtst_pred_raw - yte_raw
        dm_stat, dm_pval = dm_test(e_sota, e_patch)

    return {
        "overall_da": _da(pred_raw, yte_raw),
        "crash_da":   reg_da["Crash"],
        "boom_da":    reg_da["Boom"],
        "normal_da":  reg_da["Normal"],
        "recovery_da":reg_da["Recovery"],
        "rmse":       rmse,
        "dm_stat":    dm_stat,
        "dm_pval":    dm_pval,
        "pred_raw":   pred_raw,   # keep for potential downstream use
    }

# ═══════════════════════════════════════════════════════════════════
# SECTION 4 | MAIN LOOP
# ═══════════════════════════════════════════════════════════════════

import json
from sklearn.linear_model import LinearRegression

def run_ar5(Xtr, ytr_raw, Xte, yte_raw, rte, H, patch_pred_raw=None):
    """AR(5): OLS on last 5 log-returns (feature 0 = log_ret) of each window."""
    # Extract last 5 log-return values from each window: shape [N, 5]
    def _feats(X): return X[:, -5:, 0]   # last 5 timesteps of log_ret channel
    reg = LinearRegression().fit(_feats(Xtr), ytr_raw)
    pred_raw = reg.predict(_feats(Xte)).astype(np.float32)

    def _da(p, a, idx=None):
        if idx is not None: p, a = p[idx], a[idx]
        return float(np.mean(np.sign(p) == np.sign(a))) if len(p) >= 3 else float("nan")

    reg_da = {nm: _da(pred_raw, yte_raw, np.where(rte == r)[0])
              for r, nm in REG_NAMES.items()}
    rmse = float(np.sqrt(mean_squared_error(yte_raw, pred_raw)))
    dm_stat, dm_pval = float("nan"), float("nan")
    if patch_pred_raw is not None:
        dm_stat, dm_pval = dm_test(pred_raw - yte_raw, patch_pred_raw - yte_raw)
    return {"overall_da": _da(pred_raw, yte_raw),
            "crash_da": reg_da["Crash"], "boom_da": reg_da["Boom"],
            "normal_da": reg_da["Normal"], "recovery_da": reg_da["Recovery"],
            "rmse": rmse, "dm_stat": dm_stat, "dm_pval": dm_pval,
            "pred_raw": pred_raw, "conv_ep": 1, "train_secs": 0.0}

def run_xgboost(Xtr, ytr_raw, Xva, yva_raw, Xte, yte_raw, rte, H,
                patch_pred_raw=None, seed=42):
    """XGBoost: 100 trees, max_depth=4, flattened 42×12 window. Paper §7.1 spec."""
    try:
        from xgboost import XGBRegressor
        xgb = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1,
                           subsample=0.8, colsample_bytree=0.8,
                           random_state=seed, n_jobs=-1, verbosity=0,
                           early_stopping_rounds=20, eval_metric="mae")
        Xtr_f = Xtr.reshape(len(Xtr), -1)
        Xva_f = Xva.reshape(len(Xva), -1)
        Xte_f = Xte.reshape(len(Xte), -1)
        xgb.fit(Xtr_f, ytr_raw, eval_set=[(Xva_f, yva_raw)], verbose=False)
        pred_raw = xgb.predict(Xte_f).astype(np.float32)
        n_trees = xgb.best_iteration + 1 if hasattr(xgb, "best_iteration") else 100
    except ImportError:
        from sklearn.ensemble import GradientBoostingRegressor
        print("  xgboost not found — using sklearn GradientBoosting fallback")
        xgb = GradientBoostingRegressor(n_estimators=100, max_depth=4,
                                        learning_rate=0.1, random_state=seed)
        xgb.fit(Xtr.reshape(len(Xtr), -1), ytr_raw)
        pred_raw = xgb.predict(Xte.reshape(len(Xte), -1)).astype(np.float32)
        n_trees = 100

    def _da(p, a, idx=None):
        if idx is not None: p, a = p[idx], a[idx]
        return float(np.mean(np.sign(p) == np.sign(a))) if len(p) >= 3 else float("nan")

    reg_da = {nm: _da(pred_raw, yte_raw, np.where(rte == r)[0])
              for r, nm in REG_NAMES.items()}
    rmse = float(np.sqrt(mean_squared_error(yte_raw, pred_raw)))
    dm_stat, dm_pval = float("nan"), float("nan")
    if patch_pred_raw is not None:
        dm_stat, dm_pval = dm_test(pred_raw - yte_raw, patch_pred_raw - yte_raw)
    return {"overall_da": _da(pred_raw, yte_raw),
            "crash_da": reg_da["Crash"], "boom_da": reg_da["Boom"],
            "normal_da": reg_da["Normal"], "recovery_da": reg_da["Recovery"],
            "rmse": rmse, "dm_stat": dm_stat, "dm_pval": dm_pval,
            "pred_raw": pred_raw, "conv_ep": n_trees, "train_secs": 0.0}


def _pred_file(model_name, H):
    return os.path.join(OUT, f"pred_{model_name}_H{H}d.npy")

def _meta_file(model_name, H):
    return os.path.join(OUT, f"meta_{model_name}_H{H}d.json")

def _is_done(model_name, H):
    """True only if both prediction array and metadata JSON exist on disk."""
    return os.path.exists(_pred_file(model_name, H)) and \
           os.path.exists(_meta_file(model_name, H))

def _save_checkpoint(model, model_name, H, metrics, best_ep, train_secs, seed):
    """Atomically save everything needed to resume or skip this (model, H)."""
    # 1. Predictions array
    np.save(_pred_file(model_name, H), metrics["pred_raw"])
    # 2. Scalar metrics + metadata (no arrays in JSON)
    meta = {k: v for k, v in metrics.items()
            if k != "pred_raw" and not isinstance(v, np.ndarray)}
    meta.update({"conv_ep": best_ep, "train_secs": train_secs,
                 "seed": seed, "H": H, "model": model_name})
    with open(_meta_file(model_name, H), "w") as f:
        json.dump(meta, f, indent=2)
    # 3. Full model weights (for post-hoc inspection)
    _ckpt_path = os.path.join(OUT, f"ckpt_{model_name}_H{H}d.pt")
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, _ckpt_path)

def _load_checkpoint(model_name, H, yte_raw, rte, patch_pred_raw):
    """Load saved predictions, recompute DM test with current patch_pred_raw."""
    pred_raw = np.load(_pred_file(model_name, H))
    with open(_meta_file(model_name, H)) as f:
        meta = json.load(f)
    # Recompute DM test — patch_pred_raw may have changed if PatchTST was also resumed
    dm_stat, dm_pval = float("nan"), float("nan")
    if patch_pred_raw is not None:
        dm_stat, dm_pval = dm_test(pred_raw - yte_raw, patch_pred_raw - yte_raw)
    metrics = {**meta, "pred_raw": pred_raw,
               "dm_stat": dm_stat, "dm_pval": dm_pval}
    return metrics, meta.get("conv_ep", -1), meta.get("train_secs", 0)


ALL_MODEL_NAMES = list(MODELS.keys()) + CLASSICAL_MODELS
results     = {m: {} for m in ALL_MODEL_NAMES}
conv_epochs = {m: {} for m in ALL_MODEL_NAMES}

# Print resume status before starting
print("\n--- RESUME STATUS ---")
for H in HORIZONS:
    for m in ALL_MODEL_NAMES:
        status = "DONE (will skip)" if _is_done(m, H) else "pending"
        print(f"  H={H}d  {m:<16}  {status}")

for H in HORIZONS:
    _seed = 42 + H * 100
    torch.manual_seed(_seed); np.random.seed(_seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(_seed)

    print(f"\n{'='*60}\nH = {H}d\n{'='*60}")
    Xtr, ytr_raw, ytr_det, ttr, rtr = build_windows(df_tr, H)
    Xva, yva_raw, yva_det, tva, rva = build_windows(df_va, H)
    Xte, yte_raw, yte_det, tte, rte = build_windows(df_te, H)
    print(f"  Windows — Train:{len(Xtr)}  Val:{len(Xva)}  Test:{len(Xte)}")

    assert len(Xte) == len(yte_raw) == len(rte), "Test window count mismatch"

    patch_pred_raw = None

    for model_name, ModelClass in MODELS.items():

        # ── RESUME PATH: skip training, load from disk ──────────────
        if _is_done(model_name, H):
            metrics, best_ep, train_secs = _load_checkpoint(
                model_name, H, yte_raw, rte, patch_pred_raw)
            conv_epochs[model_name][H] = best_ep
            results[model_name][H] = metrics
            if model_name == "PatchTST":
                patch_pred_raw = metrics["pred_raw"].copy()
            print(f"  [{model_name}]  RESUMED from disk  "
                  f"(conv_ep={best_ep}  train_secs={train_secs:.0f})")
            print(f"    Overall DA={metrics['overall_da']:.3f}  "
                  f"Crash DA={metrics['crash_da']:.3f}  "
                  f"Boom DA={metrics['boom_da']:.3f}")
            continue

        # ── TRAIN PATH ───────────────────────────────────────────────
        torch.manual_seed(_seed); np.random.seed(_seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(_seed)

        model = ModelClass().to(DEVICE)
        n_params = sum(p.numel() for p in model.parameters())
        _model_lr = MODEL_LR.get(model_name, LR)
        print(f"\n  [{model_name}]  {n_params:,} params  lr={_model_lr}  TRAINING...")

        _t0 = _time.time()
        model, best_ep = train_model(model, Xtr, ytr_det, Xva, yva_det,
                                     seed=_seed, lr=_model_lr)
        _train_secs = _time.time() - _t0
        conv_epochs[model_name][H] = best_ep
        print(f"    train time: {_train_secs:.0f}s")

        metrics = evaluate(model, Xte, yte_raw, yte_det, tte, rte,
                           patchtst_pred_raw=patch_pred_raw)

        if model_name == "PatchTST":
            patch_pred_raw = metrics["pred_raw"].copy()

        results[model_name][H] = metrics

        # Save immediately — if next model crashes, this one is safe
        _save_checkpoint(model, model_name, H, metrics, best_ep, _train_secs, _seed)
        print(f"    checkpoint saved → pred_{model_name}_H{H}d.npy")

        sig = ""
        if not np.isnan(metrics["dm_pval"]):
            sig = "**" if metrics["dm_pval"] < 0.01 else ("*" if metrics["dm_pval"] < 0.05 else "ns")
        print(f"  Overall DA={metrics['overall_da']:.3f}  "
              f"Crash DA={metrics['crash_da']:.3f}  "
              f"Boom DA={metrics['boom_da']:.3f}  "
              f"Normal DA={metrics['normal_da']:.3f}  "
              f"Recovery DA={metrics['recovery_da']:.3f}")
        print(f"  RMSE={metrics['rmse']:.5f}  "
              f"DM p={metrics['dm_pval']:.4f} {sig}")

        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # ── Classical models (AR5, XGBoost) — same test split, resume-aware ──
    for cname in CLASSICAL_MODELS:
        if _is_done(cname, H):
            metrics, best_ep, _ = _load_checkpoint(cname, H, yte_raw, rte, patch_pred_raw)
            results[cname][H] = metrics
            conv_epochs[cname][H] = best_ep
            print(f"  [{cname}]  RESUMED from disk  "
                  f"Overall DA={metrics['overall_da']:.3f}  Crash DA={metrics['crash_da']:.3f}")
            continue

        _t0 = _time.time()
        if cname == "AR5":
            print(f"\n  [AR5]  OLS on last 5 log-returns")
            metrics = run_ar5(Xtr, ytr_raw, Xte, yte_raw, rte, H, patch_pred_raw)
        else:  # XGBoost
            print(f"\n  [XGBoost]  100 trees max_depth=4  flat 42×12 window")
            metrics = run_xgboost(Xtr, ytr_raw, Xva, yva_raw,
                                  Xte, yte_raw, rte, H, patch_pred_raw, seed=_seed)
        metrics["train_secs"] = _time.time() - _t0

        results[cname][H] = metrics
        conv_epochs[cname][H] = metrics.get("conv_ep", -1)

        # Reuse same checkpoint infrastructure — creates pred/meta/ckpt files
        # (no torch model to save for classical; ckpt file will be empty placeholder)
        np.save(_pred_file(cname, H), metrics["pred_raw"])
        meta = {k: v for k, v in metrics.items()
                if k != "pred_raw" and not isinstance(v, np.ndarray)}
        meta.update({"conv_ep": metrics.get("conv_ep", -1),
                     "train_secs": metrics["train_secs"],
                     "seed": _seed, "H": H, "model": cname})
        with open(_meta_file(cname, H), "w") as f:
            json.dump(meta, f, indent=2)
        # Placeholder ckpt so _is_done() returns True next run
        open(os.path.join(OUT, f"ckpt_{cname}_H{H}d.pt"), "w").close()

        print(f"  Overall DA={metrics['overall_da']:.3f}  "
              f"Crash DA={metrics['crash_da']:.3f}  "
              f"Boom DA={metrics['boom_da']:.3f}  "
              f"RMSE={metrics['rmse']:.5f}  "
              f"DM p={metrics['dm_pval']:.4f}")

    print(f"\n  H={H}d complete.")

# ═══════════════════════════════════════════════════════════════════
# SECTION 5 | RESULTS TABLE
# ═══════════════════════════════════════════════════════════════════
import io, contextlib

def _sig(pval):
    if np.isnan(pval): return "  "
    return "**" if pval < 0.01 else ("* " if pval < 0.05 else "ns")

def print_table(metric_key, title, fmt=".3f"):
    print(f"\n{'='*72}")
    print(title)
    print(f"{'='*72}")
    hdr = f"{'Model':<20}" + "".join(f"  H={H}d" for H in HORIZONS)
    print(hdr); print("-"*60)
    for m in ALL_MODEL_NAMES:
        row = f"{m:<20}"
        for H in HORIZONS:
            v = results[m].get(H, {}).get(metric_key, float("nan"))
            row += f"  {v:{fmt}}"
        pval_21 = results[m].get(21, {}).get("dm_pval", float("nan"))
        row += f"  DM(H=21d)={_sig(pval_21)}"
        print(row)

print_table("overall_da",   "TABLE 4 — OVERALL DIRECTIONAL ACCURACY")
print_table("crash_da",     "CRASH REGIME DA  (primary metric)")
print_table("boom_da",      "BOOM REGIME DA")
print_table("normal_da",    "NORMAL REGIME DA")
print_table("recovery_da",  "RECOVERY REGIME DA")
print_table("rmse",         "RMSE", fmt=".5f")

# Convergence summary
print(f"\n{'='*72}")
print("CONVERGENCE EPOCHS (early-stop)")
print(f"{'='*72}")
hdr2 = f"{'Model':<16}" + "".join(f"  H={H}d" for H in HORIZONS)
print(hdr2); print("-"*56)
for m in ALL_MODEL_NAMES:
    row = f"{m:<20}"
    for H in HORIZONS:
        row += f"  {conv_epochs[m].get(H, 0):5d}"
    print(row)

# DM test full matrix
print(f"\n{'='*72}")
print("DIEBOLD-MARIANO vs PatchTST  (p-value; ** p<0.01  * p<0.05  ns=not sig)")
print("Negative DM stat = this model is WORSE than PatchTST")
print(f"{'='*72}")
hdr3 = f"{'Model':<20}" + "".join(f"  H={H}d    " for H in HORIZONS)
print(hdr3); print("-"*72)
for m in ALL_MODEL_NAMES:
    row = f"{m:<20}"
    for H in HORIZONS:
        if m == "PatchTST":
            row += f"  {'[ref]':>8}  "
        else:
            pv = results[m][H].get("dm_pval", float("nan"))
            row += f"  p={pv:.3f}{_sig(pv)}"
    print(row)

# ── Save full report ───────────────────────────────────────────────
report_path = os.path.join(OUT, "sota_baselines_report.txt")
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    print("SOTA BASELINES REPORT — DA-MetaForecaster paper")
    print(HPARAM_SUMMARY)
    print("Data: S&P 500 2000-2023 | Train:2000-2015 | Val:2016-2017 | Test:2018-2023")
    print("Loss: MAE | Optimizer: Adam lr=1e-3 cosine | EarlyStop: patience=30 min150ep\n")
    for mk, title, fmt in [
        ("overall_da",   "OVERALL DA",    ".3f"),
        ("crash_da",     "CRASH DA",      ".3f"),
        ("boom_da",      "BOOM DA",       ".3f"),
        ("normal_da",    "NORMAL DA",     ".3f"),
        ("recovery_da",  "RECOVERY DA",   ".3f"),
        ("rmse",         "RMSE",          ".5f"),
    ]:
        print_table(mk, title, fmt)
    # Convergence
    print(f"\nCONVERGENCE EPOCHS  (AR5/XGBoost show n_trees or 1)")
    for m in ALL_MODEL_NAMES:
        print(f"  {m}: " + "  ".join(f"H={H}d→ep{conv_epochs[m].get(H,0)}" for H in HORIZONS))
    # DM table
    print(f"\nDM TEST p-values vs PatchTST")
    for m in ALL_MODEL_NAMES:
        row = f"  {m}: "
        for H in HORIZONS:
            pv = results[m][H].get("dm_pval", float("nan"))
            row += f"H={H}d p={pv:.4f}{_sig(pv)}  "
        print(row)

total_secs = _time.time() - SCRIPT_START
with contextlib.redirect_stdout(buf):
    print(f"\nTOTAL RUNTIME: {total_secs/60:.1f} min  ({total_secs:.0f}s)")
    print(f"SINGLE-RUN NOTE: Results are from 1 run per horizon with per-horizon seeds.")
    print(f"  No multi-seed averaging. Variance not estimated.")
    print(f"  To estimate variance: re-run with HORIZONS=[21] and different seed offsets.")

with open(report_path, "w", encoding="utf-8") as f:
    f.write(buf.getvalue())
print(f"\nTotal runtime: {total_secs/60:.1f} min")
print(f"Full report saved to: {report_path}")
print(f"Checkpoints saved: ckpt_{{model}}_H{{H}}d.pt for each model × horizon")
print("Paste 'Overall DA' and 'Crash DA' rows into the paper Table 4.")
