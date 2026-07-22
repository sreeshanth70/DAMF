"""
Multi-seed check of the CENTRAL claim (H=10d, primary F5_Original window),
using the actual adapt-once TTA policy (the one that produced the paper's
headline 0.373 -> 0.459 number), not the daily-re-adaptation variant in
patchtst_multiseed.py.

Why this script exists: patchtst_multiseed.py's Phase-3 TTA re-adapts the
head every active day, which is an earlier/different policy than the
adapt-once policy (freeze-and-reuse per crash episode) actually used for
every headline number in the paper (Tables 4-7, patchtst_walkforward.py /
patchtst_macro_leakfix.py). Running the wrong variant's seeds would produce
numbers that don't reconcile with the paper's own reported result for the
same nominal seed -- exactly the kind of inconsistency we're trying to
eliminate, not introduce. This script reuses patchtst_walkforward.py's
adapt-once Phase-3 logic verbatim, wrapped in a 3-seed outer loop, and
restricted to H=10d / the F5_Original fold only (the one validated central
claim), addressing the reviewer's "multiple random seeds" request directly
and correctly.

Takes real training time (3 seeds x Phase1+ANIL+TTA).
Output: seed_variance_adaptonce_H10.json with per-seed crash DA + mean/std.
See results/seed_variance_adaptonce_H10.json for the run backing the paper's
Limitations disclosure and the response-letter Comment 3 reply.
"""
import sys, json, warnings
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from collections import deque
from data_pipeline import load_dataset
warnings.filterwarnings("ignore")

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

# ── Constants (must match patchtst_walkforward.py exactly) ──────────────
H = 10
FEATURES = ["log_ret", "vix", "mom5", "mom21", "vol21", "vol_ratio", "drawdown", "vix_change",
            "yield_slope", "yield_slope_chg", "credit_spread", "credit_spread_chg"]
LOOK_BACK, PATCH_LEN, PATCH_STR = 42, 16, 8
N_PATCHES = (LOOK_BACK - PATCH_LEN) // PATCH_STR + 1
D_MODEL, N_HEADS, N_LAYERS, FFN_DIM = 128, 8, 3, 256
TRAIN_END, VAL_END, TEST_END = "2015-12-31", "2017-12-31", "2023-12-31"  # F5_Original
EPOCHS, PATIENCE, MIN_EPOCHS, BS, LR = 500, 30, 150, 32, 1e-3
N_ANIL_ITERS, ANIL_INNER_LR, ANIL_OUTER_LR = 300, 1e-2, 5e-5
ANIL_INNER_STEPS, ANIL_N_SUPPORT, ANIL_N_QUERY = 5, 16, 8
TTA_LR, TTA_STEPS, TTA_MIN_SUPPORT, TTA_BUFFER = 5e-3, 5, 10, 20
N_SEEDS = 3

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

Xtr, ytr_raw, ytr_det, rtr, _ = build_windows(df_tr, H)
Xva, yva_raw, yva_det, rva, _ = build_windows(df_va, H)
Xte, yte_raw, yte_det, rte, dte = build_windows(df_te, H)
print(f"windows: train={len(Xtr)} val={len(Xva)} test={len(Xte)}")
ttr = ytr_raw - ytr_det
tte = yte_raw - yte_det

crash_per_seed_base, crash_per_seed_full = [], []

for s_idx in range(N_SEEDS):
    _seed = 42 + H * 100 + s_idx
    torch.manual_seed(_seed); np.random.seed(_seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(_seed)
    print(f"\n{'='*50}\nSeed {s_idx+1}/{N_SEEDS} ({_seed})\n{'='*50}")

    # ── Phase 1: backbone ────────────────────────────────────────────
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
        if ep % 25 == 0:
            print(f"  [P1] ep={ep:4d} val_loss={vl:.5f} best={best_val:.5f} no_imp={no_imp}")
        if no_imp >= PATIENCE and ep >= MIN_EPOCHS:
            print(f"  [P1] Early stop ep={ep} best={best_val:.5f}")
            break
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        pred_det = model(Xte_t).cpu().numpy()
    pred_raw = pred_det + tte

    base_crash_da = float((np.sign(pred_raw[rte == 3]) == np.sign(yte_raw[rte == 3])).mean())
    print(f"  [P1] base crash DA = {base_crash_da:.4f}")

    # ── Phase 2: ANIL ────────────────────────────────────────────────
    tr_crash_idx_all = np.where(rtr == 3)[0]
    tr_boom_idx_all = np.where(rtr == 1)[0]
    has_crash = len(tr_crash_idx_all) >= ANIL_N_SUPPORT + ANIL_N_QUERY
    has_boom = len(tr_boom_idx_all) >= ANIL_N_SUPPORT + ANIL_N_QUERY

    if has_crash or has_boom:
        Xtr_crash = Xtr[tr_crash_idx_all]; ytr_crash = ytr_det[tr_crash_idx_all]
        Xtr_boom = Xtr[tr_boom_idx_all]; ytr_boom = ytr_det[tr_boom_idx_all]
        backbone_params = [p for n, p in model.named_parameters() if 'head' not in n]
        anil_opt = torch.optim.Adam(backbone_params, lr=ANIL_OUTER_LR)
        model.train()
        for it in range(N_ANIL_ITERS):
            use_crash_task = (it % 2 == 0) and has_crash
            use_boom_task = (it % 2 == 1) and has_boom
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
        print(f"  [P2] ANIL meta-trained {N_ANIL_ITERS} episodes")

    # ── Phase 3: adapt-once TTA (crash-only) ────────────────────────
    tr_crash_buf_idx = np.where(rtr == 3)[0][-TTA_BUFFER:]
    crash_buf_X = deque([Xtr[i] for i in tr_crash_buf_idx], maxlen=TTA_BUFFER)
    crash_buf_y = deque([ytr_det[i] for i in tr_crash_buf_idx], maxlen=TTA_BUFFER)
    orig_head_state = {k: v.clone() for k, v in model.head.state_dict().items()}
    pred_tta = pred_raw.copy()

    for p in model.parameters(): p.requires_grad_(False)
    for p in model.head.parameters(): p.requires_grad_(True)
    tta_head_opt = torch.optim.Adam(model.head.parameters(), lr=TTA_LR)

    prev_regime = -1
    episode_adapted = False
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
            episode_adapted = True

        if active:
            xi = torch.tensor(Xte[i:i+1], dtype=torch.float32).to(DEVICE)
            with torch.no_grad():
                rep_i = model.get_rep(xi)
                pred_tta[i] = model.head(rep_i).cpu().item() + tte[i]

        prev_regime = cur_regime

    model.head.load_state_dict(orig_head_state)
    full_crash_da = float((np.sign(pred_tta[rte == 3]) == np.sign(yte_raw[rte == 3])).mean())
    print(f"  [P3 adapt-once] full crash DA = {full_crash_da:.4f}  (delta={full_crash_da-base_crash_da:+.4f})")

    crash_per_seed_base.append(base_crash_da)
    crash_per_seed_full.append(full_crash_da)

result = {
    "H": 10, "fold": "F5_Original", "policy": "adapt-once (matches Tables 4-7)",
    "base_crash_per_seed": crash_per_seed_base,
    "base_crash_mean": float(np.mean(crash_per_seed_base)),
    "base_crash_std": float(np.std(crash_per_seed_base)),
    "full_crash_per_seed": crash_per_seed_full,
    "full_crash_mean": float(np.mean(crash_per_seed_full)),
    "full_crash_std": float(np.std(crash_per_seed_full)),
    "delta_per_seed": [f - b for f, b in zip(crash_per_seed_full, crash_per_seed_base)],
}
result["delta_mean"] = float(np.mean(result["delta_per_seed"]))
result["delta_std"] = float(np.std(result["delta_per_seed"]))

with open("seed_variance_adaptonce_H10.json", "w") as f:
    json.dump(result, f, indent=2)
print("\nDONE.")
print(json.dumps(result, indent=2))
