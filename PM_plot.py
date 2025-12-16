# ── 0. Imports ─────────────────────────────────────────────────────────────
import numpy as np, pandas as pd, matplotlib.pyplot as plt


N_RUNS = 100
# import random
# ── 1. Scaling helpers ─────────────────────────────────────────────────────
Y_nom = np.array([64., 7., 9.])
U_nom = np.array([450., 100., 2., 950.])
u_min = np.array([300.,   0., 1., 800.])
u_max = np.array([600., 200., 3., 1000.])
u_range = u_max - u_min

df = pd.read_csv("paper_machine_opt_log_scaled.csv")

# CV
def scale_cv(y_phys):      return (y_phys - Y_nom) / Y_nom
def descale_cv(y_scaled):  return Y_nom * (1.0 + y_scaled)

# MV
def scale_mv(u_phys):      return 2*(u_phys - u_min)/u_range - 1.0
def descale_mv(u_scaled):  return u_min + 0.5*u_range*(u_scaled + 1.0)
# ---------------------------------------------------------------
#  Convert scaled CVs & MVs → physical once, store as new columns
# ---------------------------------------------------------------
Y_phys = descale_cv(df[["Y1", "Y2", "Y3"]].values)          # (N,3)
U_phys = descale_mv(df[["U1", "U2", "U3", "U4"]].values)    # (N,4)

for j, col in enumerate(["Y1_phys", "Y2_phys", "Y3_phys"]):
    df[col] = Y_phys[:, j]
for j, col in enumerate(["U1_phys", "U2_phys", "U3_phys", "U4_phys"]):
    df[col] = U_phys[:, j]

# ---------------------------------------------------------------
#  Plot — physical CVs & MVs  (with units)
# ---------------------------------------------------------------
fig, axs = plt.subplots(2, 4, figsize=(14, 7), sharex=True, dpi=250)
axs = axs.flatten()

for run in range(N_RUNS):
    sub = df[df.run == run]
    axs[0].plot(sub.step, sub.Y1_phys, alpha=0.4)   # basis weight
    axs[1].plot(sub.step, sub.Y2_phys, alpha=0.4)   # ash
    axs[2].plot(sub.step, sub.Y3_phys, alpha=0.4)   # moisture
    axs[3].plot(sub.step, sub.reward,   alpha=0.4)
    axs[4].plot(sub.step, sub.U1_phys, alpha=0.4)   # stock flow
    axs[5].plot(sub.step, sub.U2_phys, alpha=0.4)   # talc flow
    axs[6].plot(sub.step, sub.U3_phys, alpha=0.4)   # steam
    axs[7].plot(sub.step, sub.U4_phys, alpha=0.4)   # machine speed

titles = ["Y1 basis weight", "Y2 ash", "Y3 moisture", "Reward",
          "U1 stock", "U2 talc", "U3 steam", "U4 speed"]
for ax, t in zip(axs, titles):
    ax.set_title(t)
    ax.grid(True)

# ---------- y‑axis labels with units --------------------------------
axs[0].set_ylabel("g m$^{-2}$")     # basis weight
axs[4].set_ylabel("L min$^{-1}$")   # stock flow (U1)

axs[1].set_ylabel("%")              # ash content
axs[5].set_ylabel("L min$^{-1}$")   # talc flow (U2)

axs[2].set_ylabel("%")              # moisture content
axs[6].set_ylabel("kg cm$^{-2}$")   # steam (U3)

axs[3].set_ylabel("Reward")         # reward (dimensionless)
axs[7].set_ylabel("m min$^{-1}$")   # machine speed (U4)

# ---------- x‑labels -------------------------------------------------
for ax in axs[4:]:
    ax.set_xlabel("Step")

fig.suptitle("Optimised control — physical CVs & MVs")
plt.tight_layout()
plt.show()