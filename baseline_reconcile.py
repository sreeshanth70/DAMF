"""
Baseline reconciliation check (3rd-round reviewer comment 2).

Table 3 (SOTA comparison) reports overall DA for the PatchTST backbone from
a standalone run in sota_baselines.py; Table 4/5 report crash-regime DA for
the "Phase 1 only" backbone from the main pipeline (patchtst_macro_leakfix.py
/ patchtst_walkforward.py). A reviewer flagged these as an unreconciled
"different seed" inconsistency between the same nominal model.

This script re-trains the PatchTST backbone under the identical protocol and
per-horizon seed formula (seed = 42 + 100*H) used everywhere else, on the
primary F5_Original split, and computes both overall DA and crash-regime DA
directly — producing a genuine same-metric comparison instead of an
explained-away discrepancy.

Result (reported in the manuscript's Table 3 note and the response letter):
crash DA came within 1.5pp of Table 4/5's values at every horizon
(0.496/0.448/0.377/0.306 vs. 0.489/0.440/0.373/0.321 for H=1/5/10/21d).
Overall DA reproduced less tightly (up to 5.5pp at H=5d) -- one of the
reasons the paper's central claim rests on the walk-forward validation
(Section 7.5) rather than any single-run number.

Uses data_pipeline.load_dataset() -- see that module for the regime-labeling
and feature-construction logic shared across all scripts in this repo.
"""
import sys, os, json
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from data_pipeline import load_dataset

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

HORIZONS = [1, 5, 10, 21]
FEATURES = ["log_ret", "vix", "mom5", "mom21", "vol21", "vol_ratio", "drawdown", "vix_change",
            "yield_slope", "yield_slope_chg", "credit_spread", "credit_spread_chg"]
LOOK_BACK, PATCH_LEN, PATCH_STR = 42, 16, 8
N_PATCHES = (LOOK_BACK - PATCH_LEN) // PATCH_STR + 1
D_MODEL, N_HEADS, N_LAYERS, FFN_DIM = 128, 8, 3, 256
TRAIN_END, VAL_END, TEST_END = "2015-12-31", "2017-12-31", "2023-12-31"
EPOCHS, PATIENCE, MIN_EPOCHS, BS, LR = 500, 30, 150, 32, 1e-3

print("Loading dataset (primary F5_Original split)...")
df = load_dataset()
df["date"] = pd.to_datetime(df["date"])
df_tr = df[df["date"] <= TRAIN_END].reset_index(drop=True)
df_va = df[(df["date"] > TRAIN_END) & (df["date"] <= VAL_END)].reset_index(drop=True)
df_te = df[(df["date"] > VAL_END) & (df["date"] <= TEST_END)].reset_index(drop=True)
print(f"train={len(df_tr)}  val={len(df_va)}  test={len(df_te)}")


class PatchTST(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Linear(PATCH_LEN, D_MODEL)
        self.pos_embed = nn.Parameter(torch.randn(1, N_PATCHES, D_MODEL) * 0.02)
        enc = nn.TransformerEncoderLayer(d_model=D_MODEL, nhead=N_HEADS, dim_feedforward=FFN_DIM,
                                          dropout=0.1, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=N_LAYERS)
        self.norm = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, 1)
        nn.init.zeros_(self.head.bias)

    def get_rep(self, x):
        B, L, C = x.shape
        xm = x.mean(1, keepdim=True); xs = x.std(1, keepdim=True) + 1e-5
        xn = ((x - xm) / xs).permute(0, 2, 1).reshape(B * C, L)
        patches = xn.unfold(1, PATCH_LEN, PATCH_STR)
        emb = self.patch_embed(patches) + self.pos_embed
        enc = self.norm(self.encoder(emb))
        return enc.mean(1).reshape(B, C, D_MODEL).mean(1)

    def forward(self, x):
        return self.head(self.get_rep(x)).squeeze(-1)


def build_windows(split, H):
    X, y_raw, y_det, reg_arr, dates = [], [], [], [], []
    vals = split[FEATURES].values.astype(np.float32)
    ret = split["log_ret"].values.astype(np.float32)
    reg = split["regime"].values.astype(int)
    dts = split["date"].values
    for i in range(LOOK_BACK, len(vals) - H + 1):
        raw_y = float(np.sum(ret[i:i + H]))
        t_val = float(np.mean(ret[i - LOOK_BACK:i])) * H
        X.append(vals[i - LOOK_BACK:i]); y_raw.append(raw_y)
        y_det.append(raw_y - t_val); reg_arr.append(int(reg[i - 1])); dates.append(dts[i])
    return (np.array(X, np.float32), np.array(y_raw, np.float32),
            np.array(y_det, np.float32), np.array(reg_arr, int), np.array(dates))


results = {}
for H in HORIZONS:
    _seed = 42 + H * 100
    torch.manual_seed(_seed); np.random.seed(_seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(_seed)

    print(f"\n{'='*50}\nH={H}d  (seed={_seed})\n{'='*50}")
    Xtr, ytr_raw, ytr_det, rtr, _ = build_windows(df_tr, H)
    Xva, yva_raw, yva_det, rva, _ = build_windows(df_va, H)
    Xte, yte_raw, yte_det, rte, dte = build_windows(df_te, H)
    print(f"windows: train={len(Xtr)} val={len(Xva)} test={len(Xte)}")

    Xtr_t = torch.tensor(Xtr); ytr_t = torch.tensor(ytr_det)
    Xva_t = torch.tensor(Xva).to(DEVICE); yva_t = torch.tensor(yva_det).to(DEVICE)
    Xte_t = torch.tensor(Xte).to(DEVICE)
    dl = DataLoader(TensorDataset(Xtr_t, ytr_t), BS, shuffle=True)

    model = PatchTST().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=500, eta_min=LR * 0.01)

    best_val, best_state, no_imp = float("inf"), None, 0
    for ep in range(1, EPOCHS + 1):
        model.train()
        for Xb, yb in dl:
            Xb = Xb.to(DEVICE); yb = yb.to(DEVICE)
            opt.zero_grad()
            F.l1_loss(model(Xb), yb).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sch.step()
        model.eval()
        with torch.no_grad():
            vl = F.l1_loss(model(Xva_t), yva_t).item()
        if vl < best_val:
            best_val = vl; best_state = {k: v.clone() for k, v in model.state_dict().items()}; no_imp = 0
        else:
            no_imp += 1
        if ep % 25 == 0 or ep <= 5:
            print(f"  ep={ep:4d} val_loss={vl:.5f} best={best_val:.5f} no_imp={no_imp}")
        if no_imp >= PATIENCE and ep >= MIN_EPOCHS:
            print(f"  Early stop ep={ep} best={best_val:.5f}")
            break

    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        pred_det = model(Xte_t).cpu().numpy()
    t_val_te = yte_raw - yte_det
    pred_raw = pred_det + t_val_te

    overall_da = float((np.sign(pred_raw) == np.sign(yte_raw)).mean())
    crash_idx = np.where(rte == 3)[0]
    crash_da = float((np.sign(pred_raw[crash_idx]) == np.sign(yte_raw[crash_idx])).mean())
    print(f"  H={H}d  overall_DA={overall_da:.3f}  crash_DA={crash_da:.3f}  n_crash={len(crash_idx)}  epochs_trained={ep}")
    results[H] = dict(overall_da=overall_da, crash_da=crash_da, n_crash=int(len(crash_idx)), epochs=ep, seed=_seed)

with open("baseline_reconcile_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nDONE — compare crash_da above to Table 4/5's Phase-1-only crash DA:")
print("  H=1d: 0.489   H=5d: 0.440   H=10d: 0.373   H=21d: 0.321")
print(json.dumps(results, indent=2))
