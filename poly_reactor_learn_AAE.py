"""
Adversarial Auto-Encoder (AAE) for the offline-RL polymer dataset
----------------------------------------------------------------
* uses the **same 10 features** you already selected
* MinMax-scales to [-1, 1]
* symmetric encoder/decoder with tanh
* latent prior 𝒩(0, I) enforced by a small discriminator
* trains 100 epochs (recon + adversarial every batch)
* saves encoder/decoder weights  →  aae_weights.pth
* saves scaler                 →  ae_scaler.pkl
* prints first-5 recon table, per-variable MSE bar, and
  Poly / T_s trajectories exactly like your original script
"""
# ── scaling constants (inputs & states) ───────────────────────────────────────
TC_MIN, TC_MAX = 280.0, 400.0                      # K
I_MIN,  I_MAX  = 0.0,   2.5                        # kg h⁻¹
S_CONC, S_TEMP = 100.0, 350.0                      # state divisors
# -------------------------------------------------------------------------- #
# 1. Imports
# -------------------------------------------------------------------------- #
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import joblib

# -------------------------------------------------------------------------- #
# 2. Load dataset & select features
# -------------------------------------------------------------------------- #
df = pd.read_csv("multi_run_dataset.csv")
feature_cols = [
    # ── scaled process states ----------------------------------------------
    "A",          # monomer-A  (scaled: kg m⁻³ / 100)
    "I",          # initiator  (scaled: kg m⁻³ / 100)
    "R",          # free radicals (mol m⁻³ / 100)
    "P",          # polymer     (scaled: kg m⁻³ / 100)
    "T",        # reactor temperature / 350

    # ── controller error signals -------------------------------------------
    "err_P",      # set-point − P   (kg m⁻³)
    "err_T",      # set-point − T   (K)

    # ── actions (0–1 scale) -------------------------------------------------
    "u_I",        # initiator-feed action (scaled 0→1)
    "u_Tc"        # coolant-temperature action (scaled 0→1)
]

X      = df[feature_cols].values.astype(np.float32)
run_id = df["run_id"].values

# -------------------------------------------------------------------------- #
# 3. Train / test split  (runs < 285  ⇒ train)
# -------------------------------------------------------------------------- #
train_mask = run_id < 95
test_mask  = run_id >= 95

X_train = X[train_mask]
X_test  = X[test_mask]

# -------------------------------------------------------------------------- #
# 4. Scale to [-1, 1]
# -------------------------------------------------------------------------- #
scaler = MinMaxScaler(feature_range=(-1, 1))
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# -------------------------------------------------------------------------- #
# 5. DataLoader
# -------------------------------------------------------------------------- #
class TensorDS(Dataset):
    def __init__(self, arr): self.x = torch.from_numpy(arr)
    def __len__(self):       return len(self.x)
    def __getitem__(self, i):return self.x[i]

train_loader = DataLoader(
    TensorDS(X_train_scaled), batch_size=256, shuffle=True
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------------------------------- #
# 6. Encoder, Decoder, Discriminator
# -------------------------------------------------------------------------- #
LATENT_DIM = 4    # keep same bottleneck size

class Encoder(nn.Module):
    def __init__(self, input_dim=9, latent_dim=LATENT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.Tanh(),
            nn.Linear(64, 16),        nn.Tanh(),
            nn.Linear(16, latent_dim)
        )
    def forward(self, x): return self.net(x)

class Decoder(nn.Module):
    def __init__(self, output_dim=9, latent_dim=LATENT_DIM):
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

enc  = Encoder().to(device)
dec  = Decoder().to(device)
disc = Discriminator().to(device)

# Xavier init
for m in list(enc.modules())+list(dec.modules())+list(disc.modules()):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

# -------------------------------------------------------------------------- #
# 7. Losses & optimisers
# -------------------------------------------------------------------------- #
recon_loss = nn.MSELoss()
bce        = nn.BCELoss()

opt_ae   = torch.optim.Adam(list(enc.parameters())+list(dec.parameters()), lr=1e-5)
opt_disc = torch.optim.Adam(disc.parameters(), lr=1e-5)

# -------------------------------------------------------------------------- #
# 8. AAE training loop
# -------------------------------------------------------------------------- #
EPOCHS = 500
for ep in range(1, EPOCHS + 1):
    enc.train(); dec.train(); disc.train()
    rec_sum, adv_sum = 0.0, 0.0

    for x in train_loader:
        x = x.to(device)

        # ---- Phase 1: reconstruction --------------------------------------
        z      = enc(x)
        x_hat  = dec(z)
        loss_r = recon_loss(x_hat, x)

        opt_ae.zero_grad()
        loss_r.backward()
        opt_ae.step()

        # ---- Phase 2: discriminator --------------------------------------
        z_real = torch.randn_like(z)                 # prior N(0,I)
        d_real = disc(z_real.detach())
        d_fake = disc(z.detach())

        lbl_real = torch.ones_like(d_real)
        lbl_fake = torch.zeros_like(d_fake)

        loss_d = bce(d_real, lbl_real) + bce(d_fake, lbl_fake)

        opt_disc.zero_grad()
        loss_d.backward()
        opt_disc.step()

        # ---- Phase 3: encoder fooling -------------------------------------
        d_fake = disc(enc(x))                        # re-evaluate
        loss_g = bce(d_fake, lbl_real)

        opt_ae.zero_grad()
        loss_g.backward()
        opt_ae.step()

        rec_sum += loss_r.item() * x.size(0)
        adv_sum += loss_g.item() * x.size(0)

    if ep % 1 == 0:
        n = len(X_train_scaled)
        print(f"Epoch {ep:>3}/{EPOCHS} | recon MSE={rec_sum/n:.6f} | adv={adv_sum/n:.4f}")

# -------------------------------------------------------------------------- #
# 9. Save weights & scaler
# -------------------------------------------------------------------------- #
torch.save({"encoder": enc.state_dict(),
            "decoder": dec.state_dict()}, "aae_weights.pth")
joblib.dump(scaler, "ae_scaler.pkl")

# -------------------------------------------------------------------------- #
# 10. Reconstruction on test set
# -------------------------------------------------------------------------- #
enc.eval(); dec.eval()
with torch.no_grad():
    X_test_tensor  = torch.from_numpy(X_test_scaled).to(device)
    X_recon_scaled = dec(enc(X_test_tensor)).cpu().numpy()

X_recon = scaler.inverse_transform(X_recon_scaled)

# -------------------------------------------------------------------------- #
# 11. First-5 reconstruction table
# -------------------------------------------------------------------------- #
first = pd.DataFrame(
    np.hstack([X_test[:5], X_recon[:5]]),
    columns=[f"orig_{c}"  for c in feature_cols] +
            [f"recon_{c}" for c in feature_cols]
)
print("\nFirst-5 test rows (original vs. reconstruction)\n")
print(first.round(4))

# -------------------------------------------------------------------------- #
# 12. Per-variable reconstruction MSE bar
# -------------------------------------------------------------------------- #
mse_vec = np.mean((X_test - X_recon) ** 2, axis=0)
plt.figure(figsize=(8, 4), dpi=200)
plt.bar(range(len(feature_cols)), mse_vec)
plt.xticks(range(len(feature_cols)), feature_cols, rotation=45, ha="right")
plt.ylabel("MSE")
plt.title("Per-variable Reconstruction MSE  (AAE)")
plt.tight_layout(); plt.show()

# -------------------------------------------------------------------------- #
# 13. Trajectory plots for P and T_s  (AAE sanity-check)
# -------------------------------------------------------------------------- #

df_test = df.loc[test_mask].reset_index(drop=True)

# ── column indices in X_recon (all scaled) ----------------------------------
idx_P   = feature_cols.index("P")        # polymer (scaled /100)
idx_Ts  = feature_cols.index("T")      # temperature /350
idx_uI  = feature_cols.index("u_I")      # action 0–1
idx_uTc = feature_cols.index("u_Tc")     # action 0–1

# ── convert RECONSTRUCTED values → physical ---------------------------------
df_test["recon_P"]   = X_recon[:, idx_P]  * S_CONC
df_test["recon_T"]   = X_recon[:, idx_Ts] * S_TEMP
df_test["recon_f_I"] = I_MIN  + X_recon[:, idx_uI]  * (I_MAX  - I_MIN)
df_test["recon_T_c"] = TC_MIN + X_recon[:, idx_uTc] * (TC_MAX - TC_MIN)

# ── convert ACTUAL (logged) scaled values → physical ------------------------
df_test["P_phys"] = df_test["P"]   * S_CONC
df_test["T_phys"] = df_test["T"] * S_TEMP
df_test["f_I"]    = I_MIN  + df_test["u_I"]  * (I_MAX  - I_MIN)
df_test["T_c"]    = TC_MIN + df_test["u_Tc"] * (TC_MAX - TC_MIN)

# ── plotting ----------------------------------------------------------------
for rid in sorted(df_test["run_id"].unique()):
    sub = df_test[df_test["run_id"] == rid]
    t   = np.arange(len(sub))

    fig, axs = plt.subplots(2, 2, figsize=(12, 7), sharex=True, dpi=200)

    # (0,0) Polymer
    axs[0,0].plot(t, sub["P_phys"],      lw=1.8, label="actual")
    axs[0,0].plot(t, sub["recon_P"],     lw=1.8, ls="--", label="recon")
    axs[0,0].set(title="Polymer", ylabel="kg m$^{-3}$"); axs[0,0].grid(True); axs[0,0].legend()

    # (0,1) Temperature
    axs[0,1].plot(t, sub["T_phys"],      lw=1.8, label="actual")
    axs[0,1].plot(t, sub["recon_T"],     lw=1.8, ls="--", label="recon")
    axs[0,1].set(title="Temperature", ylabel="K"); axs[0,1].grid(True); axs[0,1].legend()

    # (1,0) Initiator feed
    axs[1,0].plot(t, sub["f_I"],         lw=1.8, color="tab:purple", label="actual")
    axs[1,0].plot(t, sub["recon_f_I"],   lw=1.8, ls="--", color="tab:purple", label="recon")
    axs[1,0].set(title="Initiator feed", ylabel="kg h$^{-1}$"); axs[1,0].grid(True); axs[1,0].legend()

    # (1,1) Coolant temperature
    axs[1,1].plot(t, sub["T_c"],         lw=1.8, color="tab:green", label="actual")
    axs[1,1].plot(t, sub["recon_T_c"],   lw=1.8, ls="--", color="tab:green", label="recon")
    axs[1,1].set(title="Coolant temperature", ylabel="K"); axs[1,1].grid(True); axs[1,1].legend()

    for ax in axs[1,:]: ax.set_xlabel("Timestep")

    plt.suptitle(f"Run {rid}: actual vs. reconstruction (states + actions)", y=1.03)
    plt.tight_layout(); plt.show()
