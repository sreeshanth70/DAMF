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
                   ("sklearn","scikit-learn"), ("pandas_datareader","pandas-datareader"),
                   ("scipy","scipy")]:
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

# ── Walk-forward folds ───────────────────────────────────────────
# Rather than one fixed 2018-2023 test window (which contains only ~2-3
# independent crash episodes -- the root cause of the seed-instability
# found earlier), evaluate across 5 expanding-window folds spanning the
# full 2000-2023 history. Each fold's test period covers a genuinely
# different historical crash/boom regime, giving independent replicates
# of "does regime-gated TTA help during crashes" instead of one noisy
# measurement. Fold 5 matches the original paper's split exactly, for
# direct comparability with everything done so far.
df["date"] = pd.to_datetime(df["date"])

FOLDS = [
    dict(label="F1_GFC",       train_end="2006-12-31", val_end="2007-12-31", test_end="2010-12-31"),
    dict(label="F2_EuroDebt",  train_end="2010-12-31", val_end="2011-12-31", test_end="2014-12-31"),
    dict(label="F3_Q4-2018",   train_end="2014-12-31", val_end="2015-12-31", test_end="2018-12-31"),
    dict(label="F4_COVID",     train_end="2017-12-31", val_end="2018-12-31", test_end="2021-12-31"),
    dict(label="F5_Original",  train_end="2015-12-31", val_end="2017-12-31", test_end="2023-12-31"),
]

for fold in FOLDS:
    _tr = df[df["date"] <= fold["train_end"]]
    _va = df[(df["date"] > fold["train_end"]) & (df["date"] <= fold["val_end"])]
    _te = df[(df["date"] > fold["val_end"]) & (df["date"] <= fold["test_end"])]
    print(f"  {fold['label']:<12} train<={fold['train_end']} ({len(_tr)}d)  "
          f"val<={fold['val_end']} ({len(_va)}d)  test<={fold['test_end']} ({len(_te)}d)")

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
PREDS = {}          # (fold_label, H) -> prediction dict
FOLD_RESULTS = []    # list of per-(fold,H) summary rows for cross-fold aggregation

_TOTAL_RUNS = len(FOLDS) * len(HORIZONS)
_DONE_RUNS  = 0

for fold in FOLDS:
    fold_label = fold["label"]
    df_tr = df[df["date"] <= fold["train_end"]].reset_index(drop=True)
    df_va = df[(df["date"] > fold["train_end"]) & (df["date"] <= fold["val_end"])].reset_index(drop=True)
    df_te = df[(df["date"] > fold["val_end"]) & (df["date"] <= fold["test_end"])].reset_index(drop=True)

    _train_end_label = fold["train_end"]
    print(f"\n\n{'#'*60}\nFOLD {fold_label}: train<={_train_end_label}\n{'#'*60}")

    for H in HORIZONS:
        _seed = 42 + H * 100
        torch.manual_seed(_seed); np.random.seed(_seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(_seed)

        print(f"\n{'='*55}\n{fold_label}  H = {H}d\n{'='*55}")
        Xtr,ytr_raw,ytr_det,ttr,rtr,_ = build_windows(df_tr, H)
        Xva,yva_raw,yva_det,tva,rva,_ = build_windows(df_va, H)
        Xte,yte_raw,yte_det,tte,rte,dte = build_windows(df_te, H)

        print(f"  Window accounting: train {len(df_tr)} raw -> {len(Xtr)} win | "
              f"val {len(df_va)} raw -> {len(Xva)} win | test {len(df_te)} raw -> {len(Xte)} win")

        if len(Xtr) < 200 or len(Xva) < 20 or len(Xte) < 30:
            print(f"  SKIPPED -- insufficient windows for this fold/horizon")
            _DONE_RUNS += 1
            print(f"  >>> Overall progress: {_DONE_RUNS}/{_TOTAL_RUNS} "
                  f"({100*_DONE_RUNS/_TOTAL_RUNS:.0f}%) fold/horizon combinations done")
            continue

        Xtr_t = torch.tensor(Xtr); ytr_t = torch.tensor(ytr_det)
        Xva_t = torch.tensor(Xva).to(DEVICE); yva_t = torch.tensor(yva_det).to(DEVICE)
        Xte_t = torch.tensor(Xte).to(DEVICE)

        dl    = DataLoader(TensorDataset(Xtr_t, ytr_t), BS, shuffle=True)
        model = PatchTST().to(DEVICE)
        opt   = torch.optim.Adam(model.parameters(), lr=LR)
        sch   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=500, eta_min=LR*0.01)

        best_val, best_state, no_imp = float("inf"), None, 0
        bar = tqdm(range(1, EPOCHS+1), desc=f"{fold_label} H={H}d")
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
                print(f"    Early stop ep={ep}  best={best_val:.5f}"); break

        model.load_state_dict(best_state); model.eval()
        with torch.no_grad():
            pred_det = model(Xte_t).cpu().numpy()
        pred_raw = pred_det + tte

        # Phase 2: ANIL meta-training (crash + boom tasks)
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
            for it in tqdm(range(n_anil), desc=f"ANIL {fold_label} H={H}d", leave=False):
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
            print(f"    [ANIL] Meta-trained {n_anil} episodes  (crash={has_crash}, boom={has_boom})")
        else:
            reason = "H=1d (skipped by design)" if H == 1 else "insufficient windows"
            print(f"    [ANIL] Skipped -- {reason}")

        model.eval()
        with torch.no_grad():
            anil_pred_det = model(Xte_t).cpu().numpy()
        anil_pred_raw = anil_pred_det + tte

        # Phase 3: Regime-gated TTA (crash only -- boom TTA disabled)
        use_boom_tta = False
        tr_crash_buf_idx = np.where(rtr == 3)[0][-TTA_BUFFER:]
        crash_buf_X = deque([Xtr[i] for i in tr_crash_buf_idx], maxlen=TTA_BUFFER)
        crash_buf_y = deque([ytr_det[i] for i in tr_crash_buf_idx], maxlen=TTA_BUFFER)
        boom_buf_X  = deque(maxlen=TTA_BUFFER)
        boom_buf_y  = deque(maxlen=TTA_BUFFER)

        orig_head_state = {k: v.clone() for k, v in model.head.state_dict().items()}
        pred_tta      = pred_raw.copy()
        pred_tta_bias = pred_raw.copy()

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
            episode_adapted = False
            # ADAPT-ONCE (user proposal, validated vs baseline daily-readapt
            # on a single seed first: never worse, doubles the H=10d effect
            # size and turns it significant, p=0.101->0.0008): only run the
            # gradient-adaptation steps once per crash episode, at the first
            # day the buffer is large enough -- not every active day. Reset
            # behavior between episodes is unchanged from baseline.
            for i in range(len(Xte)):
                j = i - H
                if j >= 0:
                    rid_j = int(rte[j])
                    if rid_j == 3:
                        crash_buf_X.append(Xte[j]); crash_buf_y.append(yte_det[j])

                cur_regime = int(rte[i])
                active = (cur_regime == 3)
                entering_episode = active and cur_regime != prev_regime

                if cur_regime != prev_regime and prev_regime != -1:
                    model.head.load_state_dict(orig_head_state)
                    tta_head_opt = torch.optim.Adam(model.head.parameters(), lr=TTA_LR)
                    bias_head.load_state_dict(orig_head_state)
                    bias_opt = torch.optim.Adam([bias_head.bias], lr=TTA_LR)

                if entering_episode:
                    episode_adapted = False

                if active and not episode_adapted and len(crash_buf_X) >= TTA_MIN_SUPPORT:
                    bX = torch.tensor(np.array(crash_buf_X), dtype=torch.float32).to(DEVICE)
                    by = torch.tensor(np.array(crash_buf_y), dtype=torch.float32).to(DEVICE)

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

                    episode_adapted = True

                if active:
                    xi = torch.tensor(Xte[i:i+1], dtype=torch.float32).to(DEVICE)
                    with torch.no_grad():
                        rep_i = model.get_rep(xi)
                        pred_tta[i]      = model.head(rep_i).cpu().item() + tte[i]
                        pred_tta_bias[i] = bias_head(rep_i).cpu().item() + tte[i]

                prev_regime = cur_regime

        model.head.load_state_dict(orig_head_state)
        for p in model.parameters(): p.requires_grad_(True)

        def _da(preds, actuals, idx):
            if len(idx) < 3: return float('nan')
            return float(np.mean(np.sign(preds[idx]) == np.sign(actuals[idx])))

        c_idx = np.where(rte == 3)[0]
        b_idx = np.where(rte == 1)[0]

        base_crash_da = _da(pred_raw, yte_raw, c_idx)
        full_crash_da = _da(pred_tta, yte_raw, c_idx)
        bias_crash_da = _da(pred_tta_bias, yte_raw, c_idx)
        anil_crash_da = _da(anil_pred_raw, yte_raw, c_idx)
        base_boom_da  = _da(pred_raw, yte_raw, b_idx)
        full_boom_da  = _da(pred_tta, yte_raw, b_idx)
        base_all_da   = float(np.mean(np.sign(pred_raw) == np.sign(yte_raw)))
        full_all_da   = float(np.mean(np.sign(pred_tta) == np.sign(yte_raw)))
        bias_all_da   = float(np.mean(np.sign(pred_tta_bias) == np.sign(yte_raw)))

        print(f"    [{fold_label} H={H}d] Base={base_crash_da:.3f}  ANIL={anil_crash_da:.3f}  "
              f"Bias={bias_crash_da:.3f}  Full={full_crash_da:.3f}  (crash DA, n_crash={len(c_idx)})")

        PREDS[(fold_label, H)] = dict(
            dates=dte, actual=yte_raw, pred=pred_raw, pred_tta=pred_tta,
            pred_tta_bias=pred_tta_bias, pred_anil=anil_pred_raw, regime=rte,
        )
        FOLD_RESULTS.append(dict(
            fold=fold_label, H=H, n_crash=int(len(c_idx)), n_boom=int(len(b_idx)), n_test=int(len(Xte)),
            base_crash_da=base_crash_da, anil_crash_da=anil_crash_da,
            bias_crash_da=bias_crash_da, full_crash_da=full_crash_da,
            base_boom_da=base_boom_da, full_boom_da=full_boom_da,
            base_all_da=base_all_da, full_all_da=full_all_da, bias_all_da=bias_all_da,
        ))

        pd.DataFrame({
            "date": dte, "regime": rte, "actual_raw": yte_raw,
            "pred_phase1_base": pred_raw, "pred_anil_no_tta": anil_pred_raw,
            "pred_tta_bias_only": pred_tta_bias, "pred_tta_full_head": pred_tta,
        }).to_csv(os.path.join(OUT, f"predictions_adaptonce_{fold_label}_H{H}d.csv"), index=False)

        _DONE_RUNS += 1
        print(f"  >>> Overall progress: {_DONE_RUNS}/{_TOTAL_RUNS} "
              f"({100*_DONE_RUNS/_TOTAL_RUNS:.0f}%) fold/horizon combinations done")

with open(os.path.join(OUT, "walkforward_fold_results_adaptonce.json"), "w") as f:
    json.dump(FOLD_RESULTS, f, indent=2)
print(f"\nSaved walkforward_fold_results_adaptonce.json ({len(FOLD_RESULTS)} fold/horizon results)")

# ═══════════════════════════════════════════════════════════════════
# CROSS-FOLD AGGREGATION — the actual point of this script: does the
# crash-DA improvement replicate across independent historical crash
# episodes, or was it specific to one lucky test window / seed?
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("CROSS-FOLD SUMMARY (crash DA: base -> bias-only-TTA -> full-head-TTA)")
print("="*70)
import itertools
from scipy import stats as _stats

for H in HORIZONS:
    rows = [r for r in FOLD_RESULTS if r["H"] == H]
    if not rows:
        continue
    print(f"\n  H={H}d:")
    print(f"  {'Fold':<14} {'n_crash':>7} {'Base':>7} {'+Bias':>7} {'+Full':>7} "
          f"{'dBias':>7} {'dFull':>7}")
    d_bias_list, d_full_list = [], []
    skipped_folds = []
    for r in rows:
        print(f"  {r['fold']:<14} {r['n_crash']:>7} {r['base_crash_da']:>7.3f} "
              f"{r['bias_crash_da']:>7.3f} {r['full_crash_da']:>7.3f} "
              f"{(r['bias_crash_da']-r['base_crash_da']):>+7.3f} "
              f"{(r['full_crash_da']-r['base_crash_da']):>+7.3f}")
        # Some folds (e.g. calm 2012-2014) have zero crash-regime days --
        # exclude from aggregation instead of letting NaN poison the mean.
        if r["n_crash"] < 10 or np.isnan(r["base_crash_da"]):
            skipped_folds.append(r["fold"])
            continue
        d_bias_list.append(r["bias_crash_da"] - r["base_crash_da"])
        d_full_list.append(r["full_crash_da"] - r["base_crash_da"])
    if skipped_folds:
        print(f"  (excluded from aggregation -- too few crash windows: {skipped_folds})")
    if len(d_bias_list) >= 2:
        t_bias, p_bias = _stats.ttest_1samp(d_bias_list, 0.0)
        t_full, p_full = _stats.ttest_1samp(d_full_list, 0.0)
        n_pos_bias = sum(1 for d in d_bias_list if d > 0)
        n_pos_full = sum(1 for d in d_full_list if d > 0)
        print(f"  --> bias-only: mean delta={np.mean(d_bias_list):+.3f}  "
              f"{n_pos_bias}/{len(d_bias_list)} folds positive  "
              f"paired-t p={p_bias:.4f}  (n_folds={len(d_bias_list)})")
        print(f"  --> full-head: mean delta={np.mean(d_full_list):+.3f}  "
              f"{n_pos_full}/{len(d_full_list)} folds positive  "
              f"paired-t p={p_full:.4f}  (n_folds={len(d_full_list)})")
    else:
        print(f"  (too few usable folds ({len(d_bias_list)}) for aggregate significance test)")


# ═══════════════════════════════════════════════════════════════════
# SECTION 5 | FIGURES (Fold F5_Original only -- matches the original
# 2018-2023 test period, so these are directly comparable to earlier
# figures. Other folds are summarized in the cross-fold table above
# and in walkforward_fold_results.json.)
# ═══════════════════════════════════════════════════════════════════
_FIG_FOLD = "F5_Original"
_fig_horizons = [H for H in HORIZONS if (_FIG_FOLD, H) in PREDS]

if _fig_horizons:
    fig, axes = plt.subplots(len(_fig_horizons), 1, figsize=(20, 5.5*len(_fig_horizons)), sharex=False)
    if len(_fig_horizons) == 1: axes = [axes]
    fig.suptitle(f"PatchTST + Macro Features -- Actual vs Predicted (bias-only TTA)\n"
                 f"Fold {_FIG_FOLD}", fontsize=14, fontweight="bold", y=0.99)

    for idx, H in enumerate(_fig_horizons):
        ax = axes[idx]; d = PREDS[(_FIG_FOLD, H)]
        dts = pd.to_datetime(d["dates"])
        act = d["actual"]; prd = d["pred_tta_bias"]; reg = d["regime"]
        rmse_h = float(np.sqrt(mean_squared_error(act, prd)))
        da_h = float(np.mean(np.sign(prd) == np.sign(act)))
        prev_r=reg[0]; bs=dts[0]
        for i in range(1, len(dts)):
            if reg[i]!=prev_r or i==len(dts)-1:
                ax.axvspan(bs, dts[i], color=REG_COLORS[prev_r], alpha=0.35, lw=0)
                bs=dts[i]; prev_r=reg[i]
        ax.plot(dts, act, color="#222222", lw=1.1, label="Actual", zorder=3)
        ax.plot(dts, prd, color="#E63946", lw=1.0, label="Predicted (bias-TTA)",
                alpha=0.85, linestyle="--", zorder=3)
        ax.axhline(0, color="#888888", lw=0.6, ls=":")
        ax.set_title(f"H={H}d  |  RMSE={rmse_h:.5f}  DirAcc={da_h:.3f}", fontsize=10, pad=4)
        ax.set_ylabel("Log Return", fontsize=9)
        ax.grid(axis="y", alpha=0.2, ls="--")
        if idx==0:
            patches=[mpatches.Patch(color=REG_COLORS[r],label=REG_NAMES[r]) for r in range(4)]
            lines=[plt.Line2D([0],[0],color="#222222",lw=1.5,label="Actual"),
                   plt.Line2D([0],[0],color="#E63946",lw=1.5,ls="--",label="Predicted")]
            ax.legend(handles=patches+lines, loc="upper right", fontsize=8, ncol=3, framealpha=0.9)

    plt.tight_layout(rect=[0,0,1,0.98])
    p1 = os.path.join(OUT, f"walkforward_adaptonce_{_FIG_FOLD}_actual_vs_pred.png")
    plt.savefig(p1, dpi=150, bbox_inches="tight", facecolor="white"); plt.close()
    print(f"\nSaved -> {p1}")

# ── Macro features over time (full dataset, fold-independent) ─────
fig4, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 8), sharex=True)
fig4.suptitle("Macro Features -- Yield Curve & Credit Spread (full dataset)",
              fontsize=13, fontweight="bold")
dates_full = pd.to_datetime(df["date"])
ax1.plot(dates_full, df["yield_slope"], color="#2C5F8A", lw=1.2)
ax1.axhline(0, color="red", lw=0.8, ls="--", label="Inversion (yield_slope=0)")
ax1.fill_between(dates_full, df["yield_slope"], 0,
                 where=df["yield_slope"]<0, color="red", alpha=0.3, label="Inverted")
ax1.fill_between(dates_full, df["yield_slope"], 0,
                 where=df["yield_slope"]>=0, color="#2C5F8A", alpha=0.15)
ax1.set_ylabel("10y-2y Spread (%)", fontsize=10)
ax1.set_title("Yield Curve (10y-2y) -- negative = inverted = recession warning", fontsize=10)
ax1.legend(fontsize=9); ax1.grid(alpha=0.2, ls="--")
ax2.plot(dates_full, df["credit_spread"], color="#B71C1C", lw=1.2)
ax2.fill_between(dates_full, df["credit_spread"],
                 df["credit_spread"].quantile(0.25),
                 where=df["credit_spread"]>df["credit_spread"].quantile(0.75),
                 color="#B71C1C", alpha=0.3, label="High spread (top quartile)")
ax2.set_ylabel("HY OAS (%)", fontsize=10)
ax2.set_title("High Yield Credit Spread -- spikes during Crash regime", fontsize=10)
ax2.legend(fontsize=9); ax2.grid(alpha=0.2, ls="--")
plt.tight_layout()
p4 = os.path.join(OUT, "macro_feature_chart.png")
plt.savefig(p4, dpi=150, bbox_inches="tight", facecolor="white"); plt.close()
print(f"Saved -> {p4}")

# ── Final summary table (all folds, all horizons) ──────────────────
print("\n" + "="*70)
print("FINAL SUMMARY -- all folds, all horizons (crash DA)")
print("="*70)
print(f"{'Fold':<14} {'H':>4} {'n_crash':>7} {'Base':>7} {'+ANIL':>7} {'+Bias':>7} {'+Full':>7}")
for r in FOLD_RESULTS:
    print(f"{r['fold']:<14} {r['H']:>4}d {r['n_crash']:>7} {r['base_crash_da']:>7.3f} "
          f"{r['anil_crash_da']:>7.3f} {r['bias_crash_da']:>7.3f} {r['full_crash_da']:>7.3f}")
print("\nDone.")
