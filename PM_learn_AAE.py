"""
paper_machine_aae.py
====================
Adversarial Auto-Encoder for the paper-machine Monte-Carlo dataset.

* Uses runs 0 – 94   → training
* Uses runs 95 – 99  → reconstruction test (full trajectories)
* 14-dim input  = 3 CVs + 3 errors + 4 integrators + 4 MVs
* Scaling: MinMax (-1, 1) fitted on **training** data only
* Saves:
      • ae_scaler.pkl      (sklearn MinMaxScaler)
      • aae_weights.pth    (encoder & decoder state-dicts)
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0. Imports
# ─────────────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import joblib

# ---------------------------------------------------------------------------
CSV_PATH      = "paper_machine_batch_log_scaled.csv"   # <- monte-carlo log
SCALER_OUT    = "ae_scaler.pkl"
WEIGHTS_OUT   = "aae_weights.pth"
LATENT_DIM    = 4
BATCH_SIZE    = 256
EPOCHS        = 500
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ---------------------------------------------------------------------------

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load dataset & choose features
# ─────────────────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)

feature_cols = [
    # controlled outputs (CVs)
    "Y1", "Y2", "Y3",
    # error signals
    "e1", "e2", "e3",
    # PI integrator states
    "int1", "int2", "int3", "int4",
    # manipulated variables (MVs)
    "U1", "U2", "U3", "U4",
]                                             # → 14 features

X      = df[feature_cols].values.astype(np.float32)
run_id = df["run"].values

# ─────────────────────────────────────────────────────────────────────────────
# 2. Train / test split by run-id
#       • runs 0-94  → training
#       • runs 95-99 → test (reconstruction only)
# ─────────────────────────────────────────────────────────────────────────────
train_mask = run_id < 95
test_mask  = run_id >= 95

X_train = X[train_mask]
X_test  = X[test_mask]
run_test = run_id[test_mask]            # keep for plotting

# ─────────────────────────────────────────────────────────────────────────────
# 3. Scale to [-1, 1]  (fit **only** on training data)
# ─────────────────────────────────────────────────────────────────────────────
scaler       = MinMaxScaler(feature_range=(-1, 1))
X_train_s    = scaler.fit_transform(X_train)
X_test_s     = scaler.transform(X_test)
INPUT_DIM    = X_train_s.shape[1]

# save scaler now (handy for downstream inference)
joblib.dump(scaler, SCALER_OUT)

# ─────────────────────────────────────────────────────────────────────────────
# 4. DataLoader
# ─────────────────────────────────────────────────────────────────────────────
class TensorDS(Dataset):
    def __init__(self, arr): self.x = torch.from_numpy(arr)
    def __len__(self):       return len(self.x)
    def __getitem__(self, i): return self.x[i]

train_loader = DataLoader(
    TensorDS(X_train_s), batch_size=BATCH_SIZE, shuffle=True
)

# ─────────────────────────────────────────────────────────────────────────────
# 5. Network definitions
# ─────────────────────────────────────────────────────────────────────────────
class Encoder(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, latent_dim=LATENT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.Tanh(),
            nn.Linear(64, 16),        nn.Tanh(),
            nn.Linear(16, latent_dim)
        )
    def forward(self, x): return self.net(x)

class Decoder(nn.Module):
    def __init__(self, output_dim=INPUT_DIM, latent_dim=LATENT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 16), nn.Tanh(),
            nn.Linear(16, 64),         nn.Tanh(),
            nn.Linear(64, output_dim)
        )
    def forward(self, z): return self.net(z)

class Discriminator(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(32, 16),         nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(16, 1),          nn.Sigmoid()
        )
    def forward(self, z): return self.net(z)


enc, dec, disc = Encoder().to(DEVICE), Decoder().to(DEVICE), Discriminator().to(DEVICE)

# Xavier / Glorot init
for m in list(enc.modules()) + list(dec.modules()) + list(disc.modules()):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

# ─────────────────────────────────────────────────────────────────────────────
# 6. Losses & optimisers
# ─────────────────────────────────────────────────────────────────────────────
recon_loss = nn.MSELoss()
bce        = nn.BCELoss()
opt_ae     = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=1e-5)
opt_disc   = torch.optim.Adam(disc.parameters(), lr=1e-5)

# ─────────────────────────────────────────────────────────────────────────────
# 7. Training loop
# ─────────────────────────────────────────────────────────────────────────────
for ep in range(1, EPOCHS + 1):
    enc.train(); dec.train(); disc.train()
    rec_sum, adv_sum = 0.0, 0.0

    for xb in train_loader:
        xb = xb.to(DEVICE)

        # ---- Phase 1: reconstruction --------------------------------------
        z       = enc(xb)
        xb_hat  = dec(z)
        loss_r  = recon_loss(xb_hat, xb)

        opt_ae.zero_grad(); loss_r.backward(); opt_ae.step()

        # ---- Phase 2: discriminator --------------------------------------
        z_real  = torch.randn_like(z)
        d_real  = disc(z_real.detach())
        d_fake  = disc(z.detach())
        lbl_r   = torch.ones_like(d_real)
        lbl_f   = torch.zeros_like(d_fake)
        loss_d  = bce(d_real, lbl_r) + bce(d_fake, lbl_f)

        opt_disc.zero_grad(); loss_d.backward(); opt_disc.step()

        # ---- Phase 3: encoder fooling -------------------------------------
        d_fake = disc(enc(xb))
        loss_g = bce(d_fake, lbl_r)

        opt_ae.zero_grad(); loss_g.backward(); opt_ae.step()

        rec_sum += loss_r.item() * xb.size(0)
        adv_sum += loss_g.item() * xb.size(0)

    if ep % 1 == 0:
        n = len(X_train_s)
        print(f"Epoch {ep:>3}/{EPOCHS} | recon MSE={rec_sum/n:.6f} | adv={adv_sum/n:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 8. Save artefacts
# ─────────────────────────────────────────────────────────────────────────────
torch.save({"encoder": enc.state_dict(),
            "decoder": dec.state_dict()}, WEIGHTS_OUT)
print(f"\n✔  Saved weights →  {WEIGHTS_OUT}")
print(f"✔  Saved scaler  →  {SCALER_OUT}")

# ─────────────────────────────────────────────────────────────────────────────
# 9. Reconstruction on the *test* runs 95-99
# ─────────────────────────────────────────────────────────────────────────────
enc.eval(); dec.eval()
with torch.no_grad():
    X_test_t  = torch.from_numpy(X_test_s).to(DEVICE)
    X_rec_s   = dec(enc(X_test_t)).cpu().numpy()
X_rec = scaler.inverse_transform(X_rec_s)

# ─────────────────────────────────────────────────────────────────────────────
# 10. Plot trajectories for the 5 test runs
# ─────────────────────────────────────────────────────────────────────────────
test_df = df[test_mask].reset_index(drop=True)

for rid in sorted(test_df["run"].unique()):
    sub_idx = test_df["run"] == rid
    sub     = test_df[sub_idx].reset_index(drop=True)
    t       = np.arange(len(sub))

    fig, axs = plt.subplots(2, 4, figsize=(14, 6), sharex=True, dpi=150)

    # helper to plot actual vs recon
    def pair(ax, col, title):
        idx = feature_cols.index(col)
        ax.plot(t, sub[col], lw=1.6, label="actual")
        ax.plot(t, X_rec[sub_idx, idx], lw=1.6, ls="--", label="recon")
        ax.set_title(title); ax.grid(True); ax.legend()

    # Top row: Y1, Y2, Y3, reward
    pair(axs[0,0], "Y1", "Y1")
    pair(axs[0,1], "Y2", "Y2")
    pair(axs[0,2], "Y3", "Y3")
    axs[0,3].plot(t, sub["reward"], color="tab:red")
    axs[0,3].set_title("Reward"); axs[0,3].grid(True)

    # Bottom row: U1-U4
    for ax, col in zip(axs[1,:], ["U1","U2","U3","U4"]):
        pair(ax, col, col)
        ax.set_xlabel("Step")

    plt.suptitle(f"Run {rid}: reconstruction on unseen run", y=1.03)
    plt.tight_layout(); plt.show()
