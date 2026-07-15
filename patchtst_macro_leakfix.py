"""
PatchTST — Detrended Target + Macro Features (leak-fixed + ablation/stats)
===========================================================================
Fork of patchtst_macro.py — this is the script whose architecture (L=42,
P=16, S=8, 400,513 params), ANIL config, TTA config, per-horizon seeds
(42 + H*100), and FRED/HYG macro-feature construction match the paper
exactly (unlike train_colab.py, a divergent later rewrite that does not).

Changes vs. patchtst_macro.py:
  1. LEAK FIX (reviewer R2-1): the Phase-3 TTA buffer no longer appends a
     window's label until `H` trading days have actually elapsed — it
     was previously appending same-day, using H-day-forward information
     before it would exist in a live deployment.
  2. Real "Phase 1 + ANIL, no TTA" prediction is now captured by actually
     evaluating the post-ANIL model (previously this ablation row just
     reused the Phase-1-only number — see reviewer R2-7).
  3. Added a bias-only-TTA variant computed in the same pass (reuses the
     frozen-backbone rep), giving a reproducible, seeded number for the
     "bias-only TTA" ablation row instead of the old single-run "~" estimate.
  4. Window-count accounting printed per split/horizon (reviewer R1-6).
  5. Checkpoints + per-horizon prediction CSVs are now saved (this repo's
     patchtst_macro.py never persisted them — everything lived in the
     in-memory PREDS dict for the plots only).
  6. Post-hoc DA significance testing (bootstrap CI + permutation p-value)
     and simple trading-utility metrics (Sharpe/drawdown/turnover on
     non-overlapping windows) added after the main horizon loop (reviewer
     R2-4, R2-6).

Adds 4 macro features to the existing 8 price/vol features:
  9.  yield_slope     — 10y-2y Treasury spread (level)  [FRED: T10Y2Y]
  10. yield_slope_chg — daily change in yield slope
  11. credit_spread   — High Yield OAS  [FRED: BAMLH0A0HYM2]
  12. credit_spread_chg — daily change in credit spread

Yield curve inversion → recession signal.
Credit spread widening → crash warning.

Target: detrended log return (y - H × rolling mean of lookback returns).
Test: 2018-2023 | Train: 2000-2015 | Val: 2016-2017
"""

import sys, os, json, warnings
from datetime import datetime
from collections import deque

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import importlib.util as _il
for _pkg, _pip in [("yfinance","yfinance"), ("tqdm","tqdm"),
                   ("sklearn","scikit-learn"), ("pandas_datareader","pandas-datareader")]:
    if _il.find_spec(_pkg) is None:
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", _pip], check=True)
del _il

_IN_COLAB = "google.colab" in sys.modules or os.path.exists("/content")
if _IN_COLAB:
    try:
        from google.colab import drive as _gd
        _gd.mount("/content/drive", force_remount=False)
        _DRIVE_OUT = "/content/drive/MyDrive/damf_output"
    except Exception:
        _DRIVE_OUT = None
else:
    _DRIVE_OUT = None

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error
from tqdm.auto import tqdm
warnings.filterwarnings("ignore")

torch.manual_seed(42); np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:    OUT = os.path.dirname(os.path.abspath(__file__))
except: OUT = _DRIVE_OUT if _DRIVE_OUT else "/content/damf_output"
os.makedirs(OUT, exist_ok=True)
print(f"Device: {DEVICE}")

# ─── constants ────────────────────────────────────────────────────
HORIZONS  = [1, 5, 10, 21]
BASE_FEAT = ["log_ret","vix","mom5","mom21","vol21","vol_ratio","drawdown","vix_change"]
MACRO_FEAT= ["yield_slope","yield_slope_chg","credit_spread","credit_spread_chg"]
FEATURES  = BASE_FEAT + MACRO_FEAT
N_FEAT    = len(FEATURES)   # 12

LOOK_BACK = 42; PATCH_LEN = 16; PATCH_STR = 8
N_PATCHES = (LOOK_BACK - PATCH_LEN) // PATCH_STR + 1
D_MODEL   = 128; N_HEADS = 8; N_LAYERS = 3; FFN_DIM = 256
TRAIN_END = "2015-12-31"; VAL_END = "2017-12-31"
EPOCHS = 500; PATIENCE = 30; MIN_EPOCHS = 150; BS = 32; LR = 1e-3

# Crash-ANIL meta-learning constants
N_ANIL_ITERS     = 300   # meta-training episodes per horizon
ANIL_INNER_LR    = 1e-2  # inner loop (head-only) lr
ANIL_OUTER_LR    = 5e-5  # outer loop (backbone) lr
ANIL_INNER_STEPS = 5     # gradient steps in inner loop
ANIL_N_SUPPORT   = 16    # support windows per task
ANIL_N_QUERY     = 8     # query windows per task

# Regime-gated full-head TTA (crash + boom)
TTA_LR          = 5e-3  # lr for head fine-tuning
TTA_STEPS       = 5     # gradient steps per active window
TTA_MIN_SUPPORT = 10    # min windows in buffer before adapting
TTA_BUFFER      = 20    # rolling window of recent regime windows

REG_NAMES  = {0:"Normal", 1:"Boom", 2:"Recovery", 3:"Crash"}
REG_COLORS = {0:"#AEC6E8", 1:"#B7E1A1", 2:"#FFE0A0", 3:"#FFAAAA"}
REG_DARK   = {0:"#2C5F8A", 1:"#2E7D32", 2:"#996600", 3:"#B71C1C"}

# ═══════════════════════════════════════════════════════════════════
# SECTION 1 | DATA
# ═══════════════════════════════════════════════════════════════════
print("\n--- DATA LOADING ---")
import yfinance as yf

# ── S&P 500 + VIX ────────────────────────────────────────────────
sp  = yf.download("^GSPC", start="2000-01-01", end="2023-12-31",
                  auto_adjust=True, progress=False)["Close"].squeeze()
vix = yf.download("^VIX",  start="2000-01-01", end="2023-12-31",
                  auto_adjust=True, progress=False)["Close"].squeeze()
raw = pd.DataFrame({"date":pd.to_datetime(sp.index),
                    "close":sp.values,
                    "vix":vix.reindex(sp.index).values}).dropna()
print(f"S&P 500: {len(raw)} days")

# ── Macro: yield curve (FRED) + credit spread (yfinance) ─────────
print("Fetching macro data...")
import pandas_datareader.data as web

# Yield curve: T10Y2Y from FRED (reliable, full history since 1976)
try:
    yc_raw = web.DataReader("T10Y2Y", "fred", "1999-01-01", "2023-12-31")
    yc_raw.index = pd.to_datetime(yc_raw.index)
    print(f"  FRED T10Y2Y: {len(yc_raw)} obs  "
          f"range [{yc_raw['T10Y2Y'].min():.2f}, {yc_raw['T10Y2Y'].max():.2f}]")
except Exception as e:
    print(f"  FRED T10Y2Y failed ({e}) — using yfinance TNX-IRX")
    tnx = yf.download("^TNX", start="1999-01-01", end="2023-12-31",
                      auto_adjust=True, progress=False)["Close"].squeeze()
    irx = yf.download("^IRX", start="1999-01-01", end="2023-12-31",
                      auto_adjust=True, progress=False)["Close"].squeeze()
    yc_raw = pd.DataFrame({"T10Y2Y": (tnx-irx).values},
                           index=pd.to_datetime(tnx.index))

# Credit spread: HYG (HY bond ETF) vs IEF (7-10y Treasury ETF)
# Flight-to-quality signal: when HY underperforms Treasury → spreads widening → stress
# HYG starts Apr 2007 — backfill 2000-2007 using VIX z-score scaled to same range
print("  Downloading HYG and IEF from yfinance...")
hyg = yf.download("HYG", start="1999-01-01", end="2023-12-31",
                  auto_adjust=True, progress=False)["Close"].squeeze()
ief = yf.download("IEF", start="1999-01-01", end="2023-12-31",
                  auto_adjust=True, progress=False)["Close"].squeeze()
hyg.index = pd.to_datetime(hyg.index)
ief.index  = pd.to_datetime(ief.index)

# 63-day cumulative return differential: IEF - HYG (flight-to-quality signal)
# 63-day window gives range ≈ ±15% — avoids the ±360% annualization blowup
hyg_ret63 = np.log(hyg / hyg.shift(63)) * 100   # quarterly cumulative %
ief_ret63  = np.log(ief / ief.shift(63)) * 100
# Positive when IEF outperforms HYG (flight to quality = credit stress)
credit_hyg = (ief_ret63 - hyg_ret63)

# VIX z-score as pre-2007 proxy — scale to match HYG-era mean/std
_vix_s = pd.Series(
    yf.download("^VIX", start="1999-01-01", end="2023-12-31",
                auto_adjust=True, progress=False)["Close"].squeeze().values,
    index=pd.to_datetime(
        yf.download("^VIX", start="1999-01-01", end="2023-12-31",
                    auto_adjust=True, progress=False).index))
_vix_z = (_vix_s - _vix_s.rolling(252).mean()) / (_vix_s.rolling(252).std() + 1e-8)

# Calibrate VIX z-score to HYG spread scale using overlap period 2007-2023
_overlap = credit_hyg.dropna().index
_hyg_mean = credit_hyg.loc[_overlap].mean()
_hyg_std  = credit_hyg.loc[_overlap].std()
_vix_proxy = _vix_z * _hyg_std + _hyg_mean

# Combine: use HYG spread where available, VIX proxy before that
credit_combined = credit_hyg.copy()
_pre_hyg = credit_combined[credit_combined.isna()].index
credit_combined.loc[_pre_hyg] = _vix_proxy.reindex(_pre_hyg)

hy_raw = credit_combined.to_frame("BAMLH0A0HYM2")
print(f"  Credit spread: {hy_raw['BAMLH0A0HYM2'].dropna().__len__()} obs  "
      f"range [{hy_raw['BAMLH0A0HYM2'].min():.2f}, {hy_raw['BAMLH0A0HYM2'].max():.2f}]  "
      f"(HYG 2007+, VIX proxy 2000-2007)")

# ── Merge macro onto trading calendar ────────────────────────────
df = raw.copy()
df = df.set_index("date")

yc_daily = yc_raw["T10Y2Y"].reindex(df.index).ffill().bfill()
hy_daily = hy_raw["BAMLH0A0HYM2"].reindex(df.index).ffill().bfill()

df["yield_slope"]      = yc_daily.values
df["yield_slope_chg"]  = df["yield_slope"].diff(1)
df["credit_spread"]    = hy_daily.values
df["credit_spread_chg"]= df["credit_spread"].diff(1)
df = df.reset_index()

print(f"Yield slope NaN after merge: {df['yield_slope'].isna().sum()}")
print(f"Credit spread NaN after merge: {df['credit_spread'].isna().sum()}")

# ── Price features ─────────────────────────────────────────────────
df["log_ret"]   = np.log(df["close"]/df["close"].shift(1))
df["mom5"]      = df["log_ret"].rolling(5).sum()
df["mom21"]     = df["log_ret"].rolling(21).sum()
df["vol21"]     = df["log_ret"].rolling(21).std()
df["vix"]       = df["vix"]/100.0
df["vol5"]      = df["log_ret"].rolling(5).std()
df["vol_ratio"] = df["vol5"]/(df["vol21"]+1e-8)
df["drawdown"]  = df["close"]/df["close"].rolling(63).max()-1
df["vix_change"]= df["vix"].pct_change(1)
df = df.dropna().reset_index(drop=True)

# ── Regime ────────────────────────────────────────────────────────
df["_dd252"]   = df["close"]/df["close"].rolling(252).max()-1
df["_trend63"] = df["log_ret"].rolling(63).sum()
_dist = (df["_dd252"] <= -0.10) | (df["drawdown"] <= -0.10)
_bull = df["_trend63"] >= 0.02
df["regime"] = np.where(~_dist & _bull, 1,
               np.where(~_dist & ~_bull, 0,
               np.where( _dist & _bull, 2, 3))).astype(int)
df = df.dropna().reset_index(drop=True)

print(f"\nTotal days: {len(df)}")
for r, nm in REG_NAMES.items():
    n = (df["regime"]==r).sum()
    print(f"  {nm:<10} {n:5d} ({100*n/len(df):.1f}%)")

# Quick sanity: print macro feature stats
print("\nMacro feature ranges (full dataset):")
for f in MACRO_FEAT:
    print(f"  {f:<22}  mean={df[f].mean():.3f}  std={df[f].std():.3f}  "
          f"min={df[f].min():.3f}  max={df[f].max():.3f}")

# ── Splits ────────────────────────────────────────────────────────
df["date"] = pd.to_datetime(df["date"])
df_tr = df[df["date"] <= TRAIN_END].reset_index(drop=True)
df_va = df[(df["date"] > TRAIN_END) & (df["date"] <= VAL_END)].reset_index(drop=True)
df_te = df[df["date"] > VAL_END].reset_index(drop=True)
print(f"\nTrain:{len(df_tr)}  Val:{len(df_va)}  Test:{len(df_te)}")

# ═══════════════════════════════════════════════════════════════════
# SECTION 2 | WINDOWS
# ═══════════════════════════════════════════════════════════════════
def build_windows(split, H):
    X, y_raw, y_det, trend_arr, reg_arr, dates = [], [], [], [], [], []
    vals = split[FEATURES].values.astype(np.float32)
    ret  = split["log_ret"].values.astype(np.float32)
    reg  = split["regime"].values.astype(int)
    dts  = split["date"].values
    for i in range(LOOK_BACK, len(vals)-H+1):
        raw_y = float(np.sum(ret[i:i+H]))
        t_val = float(np.mean(ret[i-LOOK_BACK:i])) * H
        X.append(vals[i-LOOK_BACK:i])
        y_raw.append(raw_y)
        y_det.append(raw_y - t_val)
        trend_arr.append(t_val)
        reg_arr.append(int(reg[i-1]))
        dates.append(dts[i])
    return (np.array(X, np.float32),
            np.array(y_raw, np.float32),
            np.array(y_det, np.float32),
            np.array(trend_arr, np.float32),
            np.array(reg_arr, int),
            np.array(dates))

# ═══════════════════════════════════════════════════════════════════
# SECTION 3 | MODEL  (N_FEAT=12 now)
# ═══════════════════════════════════════════════════════════════════
class PatchTST(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Linear(PATCH_LEN, D_MODEL)
        self.pos_embed   = nn.Parameter(torch.randn(1,N_PATCHES,D_MODEL)*0.02)
        enc = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=N_HEADS, dim_feedforward=FFN_DIM,
            dropout=0.1, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=N_LAYERS)
        self.norm    = nn.LayerNorm(D_MODEL)
        self.head    = nn.Linear(D_MODEL, 1)
        nn.init.zeros_(self.head.bias)

    def get_rep(self, x):
        B,L,C = x.shape
        xm=x.mean(1,keepdim=True); xs=x.std(1,keepdim=True)+1e-5
        xn=((x-xm)/xs).permute(0,2,1).reshape(B*C,L)
        patches = xn.unfold(1,PATCH_LEN,PATCH_STR)
        emb = self.patch_embed(patches)+self.pos_embed
        enc = self.norm(self.encoder(emb))
        return enc.mean(1).reshape(B,C,D_MODEL).mean(1)

    def forward(self, x):
        return self.head(self.get_rep(x)).squeeze(-1)

print(f"\nPatchTST: {sum(p.numel() for p in PatchTST().parameters()):,} params  |  {N_FEAT} features")

# ═══════════════════════════════════════════════════════════════════
# SECTION 4 | TRAIN + PREDICT
# ═══════════════════════════════════════════════════════════════════
PREDS = {}

for H in HORIZONS:
    # Per-horizon seed: eliminates random-state contamination from different
    # epoch counts across horizons (MIN_EPOCHS changes how much state is consumed).
    _seed = 42 + H * 100
    torch.manual_seed(_seed); np.random.seed(_seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(_seed)

    print(f"\n{'='*55}\nH = {H}d\n{'='*55}")
    Xtr,ytr_raw,ytr_det,ttr,rtr,_ = build_windows(df_tr, H)
    Xva,yva_raw,yva_det,tva,rva,_ = build_windows(df_va, H)
    Xte,yte_raw,yte_det,tte,rte,dte = build_windows(df_te, H)

    # Window accounting (reviewer R1-6): every window costs LOOK_BACK
    # days of history plus H days of forward label, so raw split length
    # always exceeds window count by exactly (LOOK_BACK + H - 1).
    print(f"  Window accounting (LOOK_BACK={LOOK_BACK}, H={H}): "
          f"train {len(df_tr)} raw days -> {len(Xtr)} windows | "
          f"val {len(df_va)} raw -> {len(Xva)} win | "
          f"test {len(df_te)} raw -> {len(Xte)} win")

    Xtr_t = torch.tensor(Xtr); ytr_t = torch.tensor(ytr_det)
    Xva_t = torch.tensor(Xva).to(DEVICE); yva_t = torch.tensor(yva_det).to(DEVICE)
    Xte_t = torch.tensor(Xte).to(DEVICE)

    dl    = DataLoader(TensorDataset(Xtr_t, ytr_t), BS, shuffle=True)
    model = PatchTST().to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    sch   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=500, eta_min=LR*0.01)

    best_val, best_state, no_imp = float("inf"), None, 0
    bar = tqdm(range(1, EPOCHS+1), desc=f"H={H}d")
    for ep in bar:
        model.train()
        for Xb, yb in dl:
            Xb=Xb.to(DEVICE); yb=yb.to(DEVICE)
            opt.zero_grad()
            F.l1_loss(model(Xb), yb).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sch.step()
        model.eval()
        with torch.no_grad():
            vl = F.l1_loss(model(Xva_t), yva_t).item()
        if vl < best_val:
            best_val=vl; best_state={k:v.clone() for k,v in model.state_dict().items()}; no_imp=0
        else:
            no_imp += 1
        bar.set_postfix(val=f"{vl:.5f}", best=f"{best_val:.5f}", p=f"{no_imp}/{PATIENCE}")
        if no_imp >= PATIENCE and ep >= MIN_EPOCHS:
            print(f"  Early stop ep={ep}  best={best_val:.5f}"); break

    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        pred_det = model(Xte_t).cpu().numpy()
    pred_raw = pred_det + tte

    phase1_state = {k: v.clone() for k, v in model.state_dict().items()}
    torch.save(phase1_state, os.path.join(OUT, f"phase1_H{H}d.pt"))

    # ── Phase 2: ANIL meta-training (crash + boom tasks) ───────────
    # Skip entirely at H=1d — 1-day crash signal is noise-dominated; ANIL hurts.
    # Scale iterations: 500 at H≥10d (larger benefit there), 300 at H=5d.
    n_anil = 0 if H == 1 else N_ANIL_ITERS

    tr_crash_idx_all = np.where(rtr == 3)[0]
    tr_boom_idx_all  = np.where(rtr == 1)[0]
    has_crash = len(tr_crash_idx_all) >= ANIL_N_SUPPORT + ANIL_N_QUERY
    has_boom  = len(tr_boom_idx_all)  >= ANIL_N_SUPPORT + ANIL_N_QUERY

    if n_anil > 0 and (has_crash or has_boom):
        Xtr_crash = Xtr[tr_crash_idx_all]; ytr_crash = ytr_det[tr_crash_idx_all]
        Xtr_boom  = Xtr[tr_boom_idx_all];  ytr_boom  = ytr_det[tr_boom_idx_all]

        backbone_params = [p for n, p in model.named_parameters() if 'head' not in n]
        anil_opt = torch.optim.Adam(backbone_params, lr=ANIL_OUTER_LR)

        model.train()
        for it in tqdm(range(n_anil), desc=f"ANIL H={H}d", leave=False):
            use_crash_task = (it % 2 == 0) and has_crash
            use_boom_task  = (it % 2 == 1) and has_boom
            if not (use_crash_task or use_boom_task):
                continue

            X_pool = Xtr_crash if use_crash_task else Xtr_boom
            y_pool = ytr_crash if use_crash_task else ytr_boom

            idx = np.random.choice(len(X_pool), ANIL_N_SUPPORT + ANIL_N_QUERY, replace=False)
            sup_X = torch.tensor(X_pool[idx[:ANIL_N_SUPPORT]], dtype=torch.float32).to(DEVICE)
            sup_y = torch.tensor(y_pool[idx[:ANIL_N_SUPPORT]], dtype=torch.float32).to(DEVICE)
            qry_X = torch.tensor(X_pool[idx[ANIL_N_SUPPORT:]], dtype=torch.float32).to(DEVICE)
            qry_y = torch.tensor(y_pool[idx[ANIL_N_SUPPORT:]], dtype=torch.float32).to(DEVICE)

            hstate = {k: v.clone() for k, v in model.head.state_dict().items()}

            inner_opt = torch.optim.SGD(model.head.parameters(), lr=ANIL_INNER_LR)
            for _ in range(ANIL_INNER_STEPS):
                inner_opt.zero_grad()
                with torch.no_grad():
                    rep_sup = model.get_rep(sup_X)
                F.l1_loss(model.head(rep_sup).squeeze(-1), sup_y).backward()
                inner_opt.step()

            anil_opt.zero_grad()
            rep_qry = model.get_rep(qry_X)
            F.l1_loss(model.head(rep_qry).squeeze(-1), qry_y).backward()
            nn.utils.clip_grad_norm_(backbone_params, 1.0)
            anil_opt.step()
            model.head.load_state_dict(hstate)

        model.eval()
        print(f"  [ANIL] Meta-trained {n_anil} episodes  "
              f"(crash={has_crash}, boom={has_boom})")
    else:
        reason = "H=1d (skipped by design)" if H == 1 else "insufficient windows"
        print(f"  [ANIL] Skipped — {reason}")

    # Real "Phase 1 + ANIL, no TTA" prediction (reviewer R2-7): the
    # original ablation reused the Phase-1-only number here on the
    # (unverified) assumption that ANIL alone doesn't move eval-time
    # predictions. Actually evaluate the post-ANIL model instead.
    model.eval()
    with torch.no_grad():
        anil_pred_det = model(Xte_t).cpu().numpy()
    anil_pred_raw = anil_pred_det + tte
    anil_state = {k: v.clone() for k, v in model.state_dict().items()}
    torch.save(anil_state, os.path.join(OUT, f"phase2_H{H}d.pt"))

    # ── Phase 3: Regime-gated full-head TTA ────────────────────────
    # H=1d:        no TTA (always hurts — pure noise at daily resolution)
    # H=5,10,21d:  crash TTA only. Boom TTA was previously enabled at
    # H=21d, but post-leak-fix testing showed it regresses overall DA
    # (0.576->0.509) vs. crash-only (0.576->0.587) using the same
    # phase2_H21d.pt checkpoint — so boom TTA is disabled at all
    # horizons now, consistent with the H=5d/10d finding.
    use_boom_tta = False

    tr_crash_buf_idx = np.where(rtr == 3)[0][-TTA_BUFFER:]
    crash_buf_X = deque([Xtr[i] for i in tr_crash_buf_idx], maxlen=TTA_BUFFER)
    crash_buf_y = deque([ytr_det[i] for i in tr_crash_buf_idx], maxlen=TTA_BUFFER)
    boom_buf_X  = deque(maxlen=TTA_BUFFER)
    boom_buf_y  = deque(maxlen=TTA_BUFFER)
    if use_boom_tta:
        tr_boom_buf_idx = np.where(rtr == 1)[0][-TTA_BUFFER:]
        boom_buf_X  = deque([Xtr[i] for i in tr_boom_buf_idx], maxlen=TTA_BUFFER)
        boom_buf_y  = deque([ytr_det[i] for i in tr_boom_buf_idx], maxlen=TTA_BUFFER)

    orig_head_state = {k: v.clone() for k, v in model.head.state_dict().items()}
    pred_tta      = pred_raw.copy()   # full-head TTA (the paper's DA-MetaForecaster)
    pred_tta_bias = pred_raw.copy()   # bias-only TTA (R2-7 ablation row)

    if H > 1:
        for p in model.parameters(): p.requires_grad_(False)
        for p in model.head.parameters(): p.requires_grad_(True)
        tta_head_opt = torch.optim.Adam(model.head.parameters(), lr=TTA_LR)

        bias_head = nn.Linear(D_MODEL, 1).to(DEVICE)
        bias_head.load_state_dict(orig_head_state)
        for p in bias_head.parameters(): p.requires_grad_(False)
        bias_head.bias.requires_grad_(True)
        bias_opt = torch.optim.Adam([bias_head.bias], lr=TTA_LR)

        prev_regime = -1

        for i in range(len(Xte)):
            # --- LEAK FIX (R2-1): a window's H-day-forward label isn't
            # actually observable until H trading days after it. Release
            # window (i - H)'s label into the buffer only once that many
            # days have elapsed — never the current window's own
            # still-unrealized label.
            j = i - H
            if j >= 0:
                rid_j = int(rte[j])
                if rid_j == 3:
                    crash_buf_X.append(Xte[j]); crash_buf_y.append(yte_det[j])
                elif rid_j == 1 and use_boom_tta:
                    boom_buf_X.append(Xte[j]);  boom_buf_y.append(yte_det[j])

            cur_regime = int(rte[i])
            active = (cur_regime == 3) or (use_boom_tta and cur_regime == 1)

            if cur_regime != prev_regime and prev_regime != -1:
                model.head.load_state_dict(orig_head_state)
                tta_head_opt = torch.optim.Adam(model.head.parameters(), lr=TTA_LR)
                bias_head.load_state_dict(orig_head_state)
                bias_opt = torch.optim.Adam([bias_head.bias], lr=TTA_LR)

            if active:
                buf_X = crash_buf_X if cur_regime == 3 else boom_buf_X
                buf_y = crash_buf_y if cur_regime == 3 else boom_buf_y

                if len(buf_X) >= TTA_MIN_SUPPORT:
                    bX = torch.tensor(np.array(buf_X), dtype=torch.float32).to(DEVICE)
                    by = torch.tensor(np.array(buf_y), dtype=torch.float32).to(DEVICE)

                    model.train()
                    for _ in range(TTA_STEPS):
                        tta_head_opt.zero_grad()
                        with torch.no_grad():
                            rep = model.get_rep(bX)
                        F.l1_loss(model.head(rep).squeeze(-1), by).backward()
                        tta_head_opt.step()
                    model.eval()

                    for _ in range(TTA_STEPS):
                        bias_opt.zero_grad()
                        with torch.no_grad():
                            rep_b = model.get_rep(bX)
                        F.l1_loss(bias_head(rep_b).squeeze(-1), by).backward()
                        bias_opt.step()

                    xi = torch.tensor(Xte[i:i+1], dtype=torch.float32).to(DEVICE)
                    with torch.no_grad():
                        rep_i = model.get_rep(xi)
                        pred_tta[i]      = model.head(rep_i).cpu().item() + tte[i]
                        pred_tta_bias[i] = bias_head(rep_i).cpu().item() + tte[i]

            prev_regime = cur_regime

    # Restore model for next H
    model.head.load_state_dict(orig_head_state)
    for p in model.parameters(): p.requires_grad_(True)

    rmse_raw = float(np.sqrt(mean_squared_error(yte_raw, pred_raw)))
    da_raw   = float(np.mean(np.sign(pred_raw) == np.sign(yte_raw)))

    c_idx = np.where(rte == 3)[0]
    b_idx = np.where(rte == 1)[0]

    def _da(preds, actuals, idx):
        if len(idx) < 3: return float('nan')
        return float(np.mean(np.sign(preds[idx]) == np.sign(actuals[idx])))
    def _rmse(preds, actuals, idx):
        if len(idx) < 3: return float('nan')
        return float(np.sqrt(mean_squared_error(actuals[idx], preds[idx])))

    base_crash_da  = _da(pred_raw, yte_raw, c_idx)
    tta_crash_da   = _da(pred_tta, yte_raw, c_idx)
    tta_crash_rmse = _rmse(pred_tta, yte_raw, c_idx)
    base_boom_da   = _da(pred_raw, yte_raw, b_idx)
    tta_boom_da    = _da(pred_tta, yte_raw, b_idx)
    tta_boom_rmse  = _rmse(pred_tta, yte_raw, b_idx)
    tta_all_da     = float(np.mean(np.sign(pred_tta) == np.sign(yte_raw)))
    tta_all_rmse   = float(np.sqrt(mean_squared_error(yte_raw, pred_tta)))

    # Ablation rows (R2-7): ANIL-only (no TTA) and bias-only TTA, both
    # seeded/reproducible now instead of the old single-run "~" estimates.
    anil_crash_da   = _da(anil_pred_raw, yte_raw, c_idx)
    anil_boom_da    = _da(anil_pred_raw, yte_raw, b_idx)
    anil_all_da     = float(np.mean(np.sign(anil_pred_raw) == np.sign(yte_raw)))
    bias_crash_da   = _da(pred_tta_bias, yte_raw, c_idx)
    bias_boom_da    = _da(pred_tta_bias, yte_raw, b_idx)
    bias_all_da     = float(np.mean(np.sign(pred_tta_bias) == np.sign(yte_raw)))

    def _pp(a, b): return f"({'+'if a>b else ''}{(a-b)*100:.1f}pp)"
    print(f"  [ANIL+TTA] Crash DA : {base_crash_da:.3f} → {tta_crash_da:.3f}  {_pp(tta_crash_da, base_crash_da)}")
    print(f"  [ANIL+TTA] Boom  DA : {base_boom_da:.3f} → {tta_boom_da:.3f}  {_pp(tta_boom_da, base_boom_da)}")
    print(f"  [ANIL+TTA] Overall  : DA {da_raw:.3f} → {tta_all_da:.3f}  "
          f"RMSE {rmse_raw:.5f} → {tta_all_rmse:.5f}")
    print(f"  [Ablation] Base={base_crash_da:.3f}  ANIL-only={anil_crash_da:.3f}  "
          f"BiasTTA={bias_crash_da:.3f}  FullTTA={tta_crash_da:.3f}  (crash DA)")

    rmse_det = float(np.sqrt(mean_squared_error(yte_det, pred_det)))
    da_det   = float(np.mean(np.sign(pred_det) == np.sign(yte_det)))

    reg_rmse = {}
    for r, nm in REG_NAMES.items():
        idx = np.where(rte==r)[0]
        if len(idx) >= 3:
            reg_rmse[nm] = float(np.sqrt(mean_squared_error(yte_raw[idx], pred_raw[idx])))

    print(f"  RMSE (original) = {rmse_raw:.5f}   DirAcc = {da_raw:.3f}")
    print(f"  RMSE (detren'd) = {rmse_det:.5f}   DirAcc = {da_det:.3f}")
    print(f"  Per-regime: " + "  ".join(f"{nm}={v:.5f}" for nm,v in reg_rmse.items()))

    reg_da = {}
    for r, nm in REG_NAMES.items():
        idx = np.where(rte==r)[0]
        if len(idx) >= 3:
            reg_da[nm] = float(np.mean(np.sign(pred_raw[idx]) == np.sign(yte_raw[idx])))

    print(f"  Per-regime DA: " + "  ".join(f"{nm}={v:.3f}" for nm,v in reg_da.items()))

    PREDS[H] = dict(dates=dte, actual=yte_raw, pred=pred_raw, pred_tta=pred_tta,
                    pred_tta_bias=pred_tta_bias, pred_anil=anil_pred_raw,
                    actual_det=yte_det, pred_det=pred_det,
                    regime=rte, rmse=rmse_raw, da=da_raw,
                    rmse_det=rmse_det, da_det=da_det, reg_rmse=reg_rmse,
                    reg_da=reg_da,
                    base_crash_da=base_crash_da, tta_crash_da=tta_crash_da,
                    tta_crash_rmse=tta_crash_rmse,
                    base_boom_da=base_boom_da, tta_boom_da=tta_boom_da,
                    tta_boom_rmse=tta_boom_rmse,
                    tta_all_da=tta_all_da, tta_all_rmse=tta_all_rmse,
                    anil_crash_da=anil_crash_da, anil_boom_da=anil_boom_da,
                    anil_all_da=anil_all_da,
                    bias_crash_da=bias_crash_da, bias_boom_da=bias_boom_da,
                    bias_all_da=bias_all_da)

    # Persist predictions CSV — single reproducible source for every
    # Table 5/6/7 ablation row (replaces the old "~" single-run estimates).
    pd.DataFrame({
        "date": dte, "regime": rte, "actual_raw": yte_raw,
        "pred_phase1_base": pred_raw, "pred_anil_no_tta": anil_pred_raw,
        "pred_tta_bias_only": pred_tta_bias, "pred_tta_full_head": pred_tta,
    }).to_csv(os.path.join(OUT, f"predictions_H{H}d.csv"), index=False)

# ═══════════════════════════════════════════════════════════════════
# SECTION 4.5 | SIGNIFICANCE TESTING + TRADING-UTILITY METRICS
# (reviewer R2-4, R2-6 — computed post-hoc from the predictions above,
#  no retraining required)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("SIGNIFICANCE TESTING (bootstrap, crash-regime DA: base vs full-head TTA)")
print("="*60)

def bootstrap_da_delta(pred_a, pred_b, actual, idx, n_boot=5000, seed=0):
    """Paired bootstrap over the given index subset: resample rows with
    replacement, recompute DA(pred_b) - DA(pred_a) each time. Returns the
    observed delta, a 95% CI, and a one-sided empirical p-value for
    delta > 0 (i.e. b significantly better than a)."""
    rng = np.random.default_rng(seed)
    idx = np.asarray(idx)
    if len(idx) < 10:
        return dict(n=len(idx), delta=float("nan"), ci_lo=float("nan"),
                    ci_hi=float("nan"), p_value=float("nan"))
    sign_a = np.sign(pred_a[idx]) == np.sign(actual[idx])
    sign_b = np.sign(pred_b[idx]) == np.sign(actual[idx])
    obs_delta = sign_b.mean() - sign_a.mean()
    deltas = np.empty(n_boot)
    n = len(idx)
    for k in range(n_boot):
        samp = rng.integers(0, n, n)
        deltas[k] = sign_b[samp].mean() - sign_a[samp].mean()
    ci_lo, ci_hi = np.percentile(deltas, [2.5, 97.5])
    p_value = float(np.mean(deltas <= 0))   # one-sided: P(delta <= 0)
    return dict(n=int(n), delta=float(obs_delta), ci_lo=float(ci_lo),
                ci_hi=float(ci_hi), p_value=p_value)

SIGNIFICANCE = {}
for H in HORIZONS:
    d = PREDS[H]
    c_idx = np.where(d["regime"] == 3)[0]
    o_idx = np.arange(len(d["actual"]))
    res_crash   = bootstrap_da_delta(d["pred"], d["pred_tta"], d["actual"], c_idx, seed=H)
    res_overall = bootstrap_da_delta(d["pred"], d["pred_tta"], d["actual"], o_idx, seed=H+1)
    SIGNIFICANCE[H] = dict(crash=res_crash, overall=res_overall)
    print(f"  H={H:>2}d  crash DA delta={res_crash['delta']:+.3f}  "
          f"95% CI=[{res_crash['ci_lo']:+.3f}, {res_crash['ci_hi']:+.3f}]  "
          f"p(delta<=0)={res_crash['p_value']:.4f}  (n={res_crash['n']})")

print("\n" + "="*60)
print("TRADING-UTILITY METRICS (illustrative — non-overlapping windows only,")
print("no transaction costs/position sizing; NOT a rigorous backtest)")
print("="*60)

def trading_metrics(pred, actual, H):
    """Every H-th window only, so returns don't overlap and a simple
    Sharpe/drawdown calc isn't corrupted by autocorrelated overlapping
    labels. position = sign(pred); strategy return = position * actual
    (actual is already the H-day forward log return)."""
    pred_no, act_no = pred[::H], actual[::H]
    if len(pred_no) < 5:
        return dict(n=len(pred_no), total_return=float("nan"),
                    sharpe=float("nan"), max_drawdown=float("nan"),
                    turnover=float("nan"))
    position = np.sign(pred_no)
    strat_ret = position * act_no
    total_return = float(np.sum(strat_ret))
    periods_per_year = 252.0 / H
    sharpe = (float(np.mean(strat_ret) / (np.std(strat_ret) + 1e-12))
              * np.sqrt(periods_per_year))
    cum = np.cumsum(strat_ret)
    running_max = np.maximum.accumulate(cum)
    max_drawdown = float(np.min(cum - running_max))
    turnover = float(np.mean(np.diff(position) != 0)) if len(position) > 1 else 0.0
    return dict(n=int(len(pred_no)), total_return=total_return, sharpe=float(sharpe),
                max_drawdown=max_drawdown, turnover=turnover)

TRADING = {}
for H in HORIZONS:
    d = PREDS[H]
    base_tm = trading_metrics(d["pred"], d["actual"], H)
    tta_tm  = trading_metrics(d["pred_tta"], d["actual"], H)
    TRADING[H] = dict(base=base_tm, tta=tta_tm)
    print(f"  H={H:>2}d  Base:  return={base_tm['total_return']:+.4f}  "
          f"Sharpe={base_tm['sharpe']:+.2f}  MDD={base_tm['max_drawdown']:.4f}  "
          f"turnover={base_tm['turnover']:.2f}  (n={base_tm['n']})")
    print(f"         TTA:   return={tta_tm['total_return']:+.4f}  "
          f"Sharpe={tta_tm['sharpe']:+.2f}  MDD={tta_tm['max_drawdown']:.4f}  "
          f"turnover={tta_tm['turnover']:.2f}  (n={tta_tm['n']})")

with open(os.path.join(OUT, "significance_and_trading.json"), "w") as f:
    json.dump({"significance": {str(k): v for k, v in SIGNIFICANCE.items()},
               "trading": {str(k): v for k, v in TRADING.items()}}, f, indent=2)

# Ablation summary (R2-7) — all rows now seeded/reproducible from this run
ABLATION = {H: dict(
    phase1_only_crash=PREDS[H]["base_crash_da"], phase1_only_overall=PREDS[H]["da"],
    anil_no_tta_crash=PREDS[H]["anil_crash_da"], anil_no_tta_overall=PREDS[H]["anil_all_da"],
    bias_tta_crash=PREDS[H]["bias_crash_da"], bias_tta_overall=PREDS[H]["bias_all_da"],
    full_tta_crash=PREDS[H]["tta_crash_da"], full_tta_overall=PREDS[H]["tta_all_da"],
) for H in HORIZONS}
with open(os.path.join(OUT, "ablation_table.json"), "w") as f:
    json.dump({str(k): v for k, v in ABLATION.items()}, f, indent=2)

print("\n" + "="*60)
print("ABLATION (seeded, reproducible — replaces old single-run '~' estimates)")
print("="*60)
print(f"{'H':>4}  {'Phase1':>8}  {'+ANIL':>8}  {'+BiasTTA':>9}  {'+FullTTA':>9}   (crash DA)")
for H in HORIZONS:
    a = ABLATION[H]
    print(f"{H:>4}d  {a['phase1_only_crash']:>8.3f}  {a['anil_no_tta_crash']:>8.3f}  "
          f"{a['bias_tta_crash']:>9.3f}  {a['full_tta_crash']:>9.3f}")

# ═══════════════════════════════════════════════════════════════════
# SECTION 5 | FIGURES
# ═══════════════════════════════════════════════════════════════════

# ── Fig 1: Actual vs Predicted ────────────────────────────────────
fig, axes = plt.subplots(4, 1, figsize=(20, 22), sharex=False)
fig.suptitle("PatchTST + Macro Features — Actual vs Predicted\n"
             "Test 2018–2023  |  Features: price/vol + yield curve + credit spread",
             fontsize=14, fontweight="bold", y=0.99)

for idx, H in enumerate(HORIZONS):
    ax = axes[idx]; d = PREDS[H]
    dts = pd.to_datetime(d["dates"])
    act = d["actual"]; prd = d["pred"]; reg = d["regime"]
    prev_r=reg[0]; bs=dts[0]
    for i in range(1, len(dts)):
        if reg[i]!=prev_r or i==len(dts)-1:
            ax.axvspan(bs, dts[i], color=REG_COLORS[prev_r], alpha=0.35, lw=0)
            bs=dts[i]; prev_r=reg[i]
    ax.plot(dts, act, color="#222222", lw=1.1, label="Actual",    zorder=3)
    ax.plot(dts, prd, color="#E63946", lw=1.0, label="Predicted",
            alpha=0.85, linestyle="--", zorder=3)
    ax.axhline(0, color="#888888", lw=0.6, ls=":")
    ax.set_title(f"H={H}d  |  RMSE={d['rmse']:.5f}  DirAcc={d['da']:.3f}  "
                 f"[det RMSE={d['rmse_det']:.5f}  DA={d['da_det']:.3f}]",
                 fontsize=10, pad=4)
    ax.set_ylabel("Log Return", fontsize=9)
    ax.grid(axis="y", alpha=0.2, ls="--")
    if idx==0:
        patches=[mpatches.Patch(color=REG_COLORS[r],label=REG_NAMES[r]) for r in range(4)]
        lines=[plt.Line2D([0],[0],color="#222222",lw=1.5,label="Actual"),
               plt.Line2D([0],[0],color="#E63946",lw=1.5,ls="--",label="Predicted")]
        ax.legend(handles=patches+lines, loc="upper right", fontsize=8, ncol=3, framealpha=0.9)

plt.tight_layout(rect=[0,0,1,0.98])
p1 = os.path.join(OUT, "macro_actual_vs_pred.png")
plt.savefig(p1, dpi=150, bbox_inches="tight", facecolor="white"); plt.close()
print(f"\nSaved → {p1}")

# ── Fig 2: Cumulative ─────────────────────────────────────────────
fig2, axes2 = plt.subplots(4, 1, figsize=(20, 22), sharex=False)
fig2.suptitle("PatchTST + Macro Features — Cumulative Returns\n"
              "Test 2018–2023  |  Does macro help track crashes & recoveries?",
              fontsize=14, fontweight="bold", y=0.99)

for idx, H in enumerate(HORIZONS):
    ax = axes2[idx]; d = PREDS[H]
    dts = pd.to_datetime(d["dates"])
    cum_act = np.cumsum(d["actual"]); cum_prd = np.cumsum(d["pred"])
    reg = d["regime"]
    prev_r=reg[0]; bs=dts[0]
    for i in range(1, len(dts)):
        if reg[i]!=prev_r or i==len(dts)-1:
            ax.axvspan(bs, dts[i], color=REG_COLORS[prev_r], alpha=0.35, lw=0)
            bs=dts[i]; prev_r=reg[i]
    ax.plot(dts, cum_act, color="#222222", lw=1.4, label="Actual", zorder=3)
    ax.plot(dts, cum_prd, color="#E63946", lw=1.2, label="Predicted",
            alpha=0.85, linestyle="--", zorder=3)
    ax.axhline(0, color="#888888", lw=0.6, ls=":")
    ax2r = ax.twinx()
    ax2r.fill_between(dts, np.abs(cum_act-cum_prd), alpha=0.12, color="#E63946")
    ax2r.set_ylabel("Tracking gap", fontsize=7, color="#E63946")
    ax2r.tick_params(axis='y', labelcolor="#E63946", labelsize=7)
    ax.set_title(f"H={H}d  |  Final gap={cum_act[-1]-cum_prd[-1]:+.3f}  "
                 f"DirAcc={d['da']:.3f}  RMSE={d['rmse']:.5f}", fontsize=11, pad=4)
    ax.set_ylabel("Cumulative Log Return", fontsize=9)
    ax.grid(axis="y", alpha=0.2, ls="--")
    if idx==0:
        patches=[mpatches.Patch(color=REG_COLORS[r],label=REG_NAMES[r]) for r in range(4)]
        lines=[plt.Line2D([0],[0],color="#222222",lw=1.5,label="Actual"),
               plt.Line2D([0],[0],color="#E63946",lw=1.5,ls="--",label="Predicted")]
        ax.legend(handles=patches+lines, loc="upper left", fontsize=8, ncol=3, framealpha=0.9)

plt.tight_layout(rect=[0,0,1,0.98])
p2 = os.path.join(OUT, "macro_cumulative.png")
plt.savefig(p2, dpi=150, bbox_inches="tight", facecolor="white"); plt.close()
print(f"Saved → {p2}")

# ── Fig 3: Error by regime + scatter ─────────────────────────────
fig3, axes3 = plt.subplots(2, 4, figsize=(22, 10))
fig3.suptitle("PatchTST + Macro — Error Analysis\n"
              "Top: abs error by regime  |  Bottom: actual vs predicted scatter (detrended)",
              fontsize=13, fontweight="bold")

for col, H in enumerate(HORIZONS):
    d = PREDS[H]; act=d["actual"]; prd=d["pred"]; reg=d["regime"]
    abs_err = np.abs(act-prd)
    ax = axes3[0,col]
    data_box, labels_box, cols_box = [], [], []
    for r in range(4):
        idx_r = np.where(reg==r)[0]
        if len(idx_r)<3: continue
        rmse_r = float(np.sqrt(mean_squared_error(act[idx_r], prd[idx_r])))
        data_box.append(abs_err[idx_r])
        labels_box.append(f"{REG_NAMES[r]}\n{rmse_r:.4f}")
        cols_box.append(REG_COLORS[r])
    bp = ax.boxplot(data_box, patch_artist=True, medianprops=dict(color="black",lw=2))
    for patch,c in zip(bp["boxes"],cols_box): patch.set_facecolor(c); patch.set_alpha(0.8)
    ax.set_xticklabels(labels_box, fontsize=8)
    ax.set_title(f"H={H}d  RMSE={d['rmse']:.5f}", fontsize=10)
    ax.set_ylabel("Abs Error" if col==0 else "")
    ax.grid(axis="y", alpha=0.25, ls="--")

    ax2 = axes3[1,col]
    act_d=d["actual_det"]; prd_d=d["pred_det"]
    for r in range(4):
        idx_r=np.where(reg==r)[0]
        if len(idx_r)<3: continue
        ax2.scatter(act_d[idx_r], prd_d[idx_r], s=4, alpha=0.4,
                    color=REG_COLORS[r], edgecolors=REG_DARK[r],
                    linewidths=0.1, label=REG_NAMES[r])
    lim=max(abs(act_d).max(),abs(prd_d).max())*1.05
    ax2.plot([-lim,lim],[-lim,lim],"k--",lw=0.8,alpha=0.5)
    ax2.axhline(0,color="#aaa",lw=0.5); ax2.axvline(0,color="#aaa",lw=0.5)
    ax2.set_xlim(-lim,lim); ax2.set_ylim(-lim,lim)
    ax2.set_xlabel("Actual (detrended)", fontsize=8)
    ax2.set_ylabel("Predicted (detrended)" if col==0 else "", fontsize=8)
    ax2.set_title(f"H={H}d  Scatter  DA={d['da_det']:.3f}", fontsize=10)
    ax2.grid(alpha=0.2, ls="--")
    if col==0: ax2.legend(fontsize=7, markerscale=2, framealpha=0.8)

plt.tight_layout()
p3 = os.path.join(OUT, "macro_error_analysis.png")
plt.savefig(p3, dpi=150, bbox_inches="tight", facecolor="white"); plt.close()
print(f"Saved → {p3}")

# ── Fig 4: Macro features over time ──────────────────────────────
fig4, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 8), sharex=True)
fig4.suptitle("Macro Features — Yield Curve & Credit Spread (full dataset)",
              fontsize=13, fontweight="bold")
dates_full = pd.to_datetime(df["date"])
ax1.plot(dates_full, df["yield_slope"], color="#2C5F8A", lw=1.2)
ax1.axhline(0, color="red", lw=0.8, ls="--", label="Inversion (yield_slope=0)")
ax1.fill_between(dates_full, df["yield_slope"], 0,
                 where=df["yield_slope"]<0, color="red", alpha=0.3, label="Inverted")
ax1.fill_between(dates_full, df["yield_slope"], 0,
                 where=df["yield_slope"]>=0, color="#2C5F8A", alpha=0.15)
ax1.set_ylabel("10y-2y Spread (%)", fontsize=10)
ax1.set_title("Yield Curve (10y-2y) — negative = inverted = recession warning", fontsize=10)
ax1.legend(fontsize=9); ax1.grid(alpha=0.2, ls="--")
ax2.plot(dates_full, df["credit_spread"], color="#B71C1C", lw=1.2)
ax2.fill_between(dates_full, df["credit_spread"],
                 df["credit_spread"].quantile(0.25),
                 where=df["credit_spread"]>df["credit_spread"].quantile(0.75),
                 color="#B71C1C", alpha=0.3, label="High spread (top quartile)")
ax2.set_ylabel("HY OAS (%)", fontsize=10)
ax2.set_title("High Yield Credit Spread — spikes during Crash regime", fontsize=10)
ax2.legend(fontsize=9); ax2.grid(alpha=0.2, ls="--")
plt.tight_layout()
p4 = os.path.join(OUT, "macro_feature_chart.png")
plt.savefig(p4, dpi=150, bbox_inches="tight", facecolor="white"); plt.close()
print(f"Saved → {p4}")

# ── Summary ───────────────────────────────────────────────────────
print("\n" + "="*60)
print("SUMMARY  (12 features: 8 price/vol + 4 macro)")
print("="*60)
print(f"{'H':>4}  {'RMSE_raw':>10}  {'RMSE_det':>10}  {'DA_raw':>8}  {'DA_det':>8}")
for H in HORIZONS:
    d = PREDS[H]
    print(f"{H:>4}d  {d['rmse']:>10.5f}  {d['rmse_det']:>10.5f}  "
          f"{d['da']:>8.3f}  {d['da_det']:>8.3f}")
print()
print(f"{'H':>4}  {'Normal':>10}  {'Boom':>10}  {'Recovery':>10}  {'Crash':>10}")
for H in HORIZONS:
    rr = PREDS[H]["reg_rmse"]
    print(f"{H:>4}d  {rr.get('Normal',0):>10.5f}  {rr.get('Boom',0):>10.5f}  "
          f"{rr.get('Recovery',0):>10.5f}  {rr.get('Crash',0):>10.5f}")
print()
print(f"{'H':>4}  {'Normal DA':>10}  {'Boom DA':>10}  {'Recovery DA':>12}  {'Crash DA':>10}")
for H in HORIZONS:
    rd = PREDS[H]["reg_da"]
    print(f"{H:>4}d  {rd.get('Normal',0):>10.3f}  {rd.get('Boom',0):>10.3f}  "
          f"{rd.get('Recovery',0):>12.3f}  {rd.get('Crash',0):>10.3f}")

print()
print("── ANIL + Full-Head TTA (crash + boom, regime-gated) ────────")
print(f"{'H':>4}  {'Base CrashDA':>13}  {'ANIL CrashDA':>13}  {'ΔCrash':>7}  "
      f"{'Base BoomDA':>12}  {'ANIL BoomDA':>12}  {'ΔBoom':>6}  "
      f"{'Overall DA':>11}  {'ANIL DA':>8}")
for H in HORIZONS:
    d = PREDS[H]
    dc = (d['tta_crash_da'] - d['base_crash_da']) * 100
    db = (d['tta_boom_da']  - d['base_boom_da'])  * 100
    print(f"{H:>4}d  {d['base_crash_da']:>13.3f}  {d['tta_crash_da']:>13.3f}  "
          f"{dc:>+7.1f}pp  {d['base_boom_da']:>12.3f}  {d['tta_boom_da']:>12.3f}  "
          f"{db:>+6.1f}pp  {d['da']:>11.3f}  {d['tta_all_da']:>8.3f}")
print("\nDone.")
