"""
Optimising control with learned Q‑function + AAE penalty
(working entirely in *scaled* CV and MV space)

Outputs
-------
• paper_machine_opt_log_scaled.csv   – full trajectory (scaled values only)
• Summary plot:  CVs+Reward (top), MVs (bottom), all runs overlayed
"""

# ── 0. Imports ─────────────────────────────────────────────────────────────
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import minimize
import joblib, torch, torch.nn as nn

CSV_OUT = "paper_machine_opt_log_scaled.csv"

# ── 1. Scaling helpers ─────────────────────────────────────────────────────
Y_nom = np.array([64., 7., 9.])
U_nom = np.array([450., 100., 2., 950.])
u_min = np.array([300.,   0., 1., 800.])
u_max = np.array([600., 200., 3., 1000.])
u_range = u_max - u_min

# CV
def scale_cv(y_phys):      return (y_phys - Y_nom) / Y_nom
def descale_cv(y_scaled):  return Y_nom * (1.0 + y_scaled)

# MV
def scale_mv(u_phys):      return 2*(u_phys - u_min)/u_range - 1.0
def descale_mv(u_scaled):  return u_min + 0.5*u_range*(u_scaled + 1.0)

u_scaled_nom = scale_mv(U_nom)                 # (‑1…+1) nominal
Ysp_scaled   = scale_cv(np.array([102., 17., 9.]))


# ============================================================================
# 2) State‑space (fill in!)
# ============================================================================
A = np.array([
    [ 0.676 ,  0.349 , -0.153 , -2.462 , -2.182 ,  0.735 , -0.477 , -1.326 ,  0.201 , -0.199 ],
    [-1.309 ,  2.494 , -0.601 , -10.70 , -3.071 , -0.850 , -0.803 , -3.458 ,  0.641 , -0.372 ],
    [ 0.941 , -1.215 ,  1.472 ,  9.160 ,  4.575 , -3.024 ,  1.526 ,  4.882 , -1.176 ,  0.902 ],
    [-0.305 ,  0.362 , -0.132 , -1.566 , -0.493 , -0.145 , -0.079 , -0.553 ,  0.151 , -0.025 ],
    [-0.043 ,  0.077 , -0.029 , -0.612 ,  0.6193,  0.449 ,  0.164 , -0.017 , -0.098 ,  0.035 ],
    [-0.318 ,  0.358 , -0.108 , -2.318 ,  0.066 ,  0.193 , -0.523 , -0.131 ,  0.146 ,  0.064 ],
    [-0.027 ,  0.071 , -0.021 , -0.491 ,  0.052 ,  0.398 ,  0.542 , -0.111 ,  0.434 , -0.060 ],
    [ 0.179 , -0.203 ,  0.024 ,  1.140 , -0.029 , -0.006 , -0.046 , -0.143 ,  0.162 ,  0.023 ],
    [ 0.437 , -0.511 ,  0.161 ,  3.481 ,  0.593 ,  0.311 , -0.526 , -0.215 ,  0.584 ,  0.220 ],
    [ 0.420 , -0.519 ,  0.174 ,  3.604 ,  0.577 ,  0.241 ,  0.102 ,  0.272 , -0.462 ,  0.354 ]
    ])
B = np.array([
    [-5.15e-03, -1.977e-02,  7.755e-01, -1.381e-02],
    [-1.938e-02, -5.644e-02,  3.630e+00, -5.918e-02],
    [-1.642e-02,  3.865e-02, -5.436e-01,  5.292e-02],
    [-2.95e-03, -7.79e-03,  4.429e-01, -1.406e-02],
    [ 7.13e-03, -8.50e-04, -6.990e-01, -4.00e-03],
    [-1.25e-02,  3.40e-04,  8.226e-01, -1.156e-02],
    [ 5.65e-03,  1.84e-03, -2.227e-01, -3.76e-03],
    [ 7.23e-03, -1.536e-02,  9.547e-01,  4.46e-03],
    [ 6.13e-03,  8.90e-04,  2.952e-01,  1.798e-02],
    [ 1.09e-03,  4.58e-03,  4.471e-02,  1.986e-02]
    ])
C = np.array([
    [-1.742 ,  0.390 ,  0.118 ,  2.347 , -2.043 ,  1.614 , -0.316 ,  0.125 ,  0.037 , -0.141 ],
    [-0.304 , -0.088 , -0.159 , -0.141 , -0.419 , -0.070 ,  0.064 , -0.355 , -0.106 ,  0.132 ],
    [-0.207 ,  0.189 ,  0.002 , -2.854 , -0.934 ,  0.275 , -0.256 , -0.822 ,  0.159 , -0.186 ]
    ])
# ── 3. Hyper‑parameters ───────────────────────────────────────────────────
N_RUNS  = 100
N_STEPS = 250
SIGMA   = 0.2


w_Q, w_AE, w_dU     = 0.5313, 0.05225, 11.04
w_Q_st, w_AE_st, w_dU_st =  0.5313, 0.05225, 15.04
ERR_BAND  = np.array([3, 0.4, 0.8]) / Y_nom    # convert band to *scaled*

# ── 4. Load models (they were trained on *scaled* data) ───────────────────
Q_model   = joblib.load("q_discounted.joblib")
q_scaler  = joblib.load("q_scaler.joblib")

ae_sd     = torch.load("aae_weights.pth", map_location="cpu")
ae_scaler = joblib.load("ae_scaler.pkl")

class Encoder(nn.Module):
    def __init__(self, input_dim=14, latent_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.Tanh(),
            nn.Linear(64, 16),        nn.Tanh(),
            nn.Linear(16, latent_dim)
        )
    def forward(self, x): return self.net(x)

class Decoder(nn.Module):
    def __init__(self, output_dim=14, latent_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 16), nn.Tanh(),
            nn.Linear(16, 64),         nn.Tanh(),
            nn.Linear(64, output_dim)
        )
    def forward(self, z): return self.net(z)

enc, dec = Encoder(), Decoder()
enc.load_state_dict(ae_sd["encoder"]); dec.load_state_dict(ae_sd["decoder"])
enc.eval(); dec.eval()

def Q_hat(sa_scaled):
    sa_scaled = q_scaler.transform(sa_scaled.reshape(1,-1)).astype(np.float32)
    return float(Q_model.predict(sa_scaled)[0])

def recon_mse(sa_scaled):
    with torch.no_grad():
        x_s = torch.from_numpy(ae_scaler.transform(sa_scaled.reshape(1,-1))).float()
        x_rec = dec(enc(x_s))
        return torch.mean((x_s - x_rec)**2).item()

# ── 5. Simulation loop ────────────────────────────────────────────────────
records = []

for run in range(N_RUNS):
    print(f"Run {run+1}/{N_RUNS}")
    x = np.random.uniform(-0.1, 0.1, size=10)
    int_term = np.zeros(4)
    u_prev   = u_scaled_nom.copy()          # scaled
    prev_err_metric = None

    for k in range(N_STEPS):
        # --- plant measurement -------------------------------------------
        y_dev_phys = C @ x
        y_phys     = (Y_nom + y_dev_phys) * np.random.uniform(0.97, 1.03, 3)
        y_scaled   = scale_cv(y_phys)

        err_scaled = Ysp_scaled - y_scaled
        abs_err    = np.abs(err_scaled)
        print(f"step {k+1}/{N_STEPS} | abs‑err (scaled):"
              f" {abs_err[0]:.3f} {abs_err[1]:.3f} {abs_err[2]:.3f}")

        # --- choose weight set -------------------------------------------
        in_band      = np.all(abs_err <= ERR_BAND)
        near_horizon = k > 235                 # ← NEW
        
        if in_band or near_horizon:
            wQ, wAE, wDU = w_Q_st, w_AE_st, w_dU_st
            # optional debug:
            # print("steady weights (", "band" if in_band else "late", ")")
        else:
            wQ, wAE, wDU = w_Q,    w_AE,    w_dU

        # --- integrators (scaled) ----------------------------------------
        int_term += np.array([err_scaled[0], err_scaled[1],
                              err_scaled[2], err_scaled[0]])

        # --- optimisation objective --------------------------------------
        def obj(u_vec):
            u_clip = np.clip(u_vec, -1.0, 1.0)
            sa = np.hstack([y_scaled, err_scaled, int_term, u_clip])
            J_core   = -wQ * Q_hat(sa) + wAE * recon_mse(sa)
            dU       = u_clip - u_prev
            J_smooth = wDU * np.sum(dU**2)
            return J_core + J_smooth

        def solve(method, x0):
            res = minimize(obj, x0, method=method,
                           options={"maxiter": 800, "disp": False})
            u = np.clip(res.x, -1.0, 1.0)
            return u, obj(u)

        u_nm, J_nm = solve("Nelder-Mead", u_prev)
        u_pw, J_pw = solve("Powell", u_prev)

        u_opt = u_nm if J_nm <= J_pw else u_pw
        u_prev = u_opt.copy()
                
        # # ── optimiser: Differential‑Evolution, bounds ±1 ─────────────────────────
        # from scipy.optimize import differential_evolution
        
        # def obj(u_vec):
        #     u = np.clip(u_vec, -1, 1)
        #     sa = np.hstack([y_scaled, err_scaled, int_term, u])
        #     return (-w_Q * Q_hat(sa)
        #             + w_AE * recon_mse(sa)
        #             + w_dU * np.sum((u - u_prev)**2))
        
        # bounds = [(-1.0, 1.0)] * 4
        
        # # SciPy ≥ 1.7 lets you fix the first population member with x0
        # result = differential_evolution(
        #     obj,
        #     bounds=bounds,
        #     x0=u_prev,          # warm‑start: first individual = last action
        #     popsize=16,          # small pop  ⇒ faster
        #     maxiter=120,         # generations (tune as needed)
        #     mutation=(0.5, 1.0),
        #     recombination=0.7,
        #     polish=True,        # quick L‑BFGS‑B finish inside bounds
        #     tol=1e-4,
        #     disp=False,
        #     updating="deferred",
        #     workers=1           # set >1 for parallel objective calls
        # )
        
        # u_opt = np.clip(result.x, -1, 1)
        # u_prev = u_opt.copy()   # warm‑start for next MPC step


        # --- anti‑windup (scaled) ----------------------------------------
        for i in range(4):
            e = err_scaled[0] if i in (0,3) else err_scaled[i-1]
            sat_lo = u_opt[i] <= -1.0 and e < 0
            sat_hi = u_opt[i] >=  1.0 and e > 0
            if not (sat_lo or sat_hi):
                pass  # integrator already updated above

        #------------------------------------------------------------------
        #optimisation OR feedback block
        #------------------------------------------------------------------

        # --- state update (needs physical Δu) ----------------------------
        delta_u_phys = descale_mv(u_opt) - U_nom
        x = A @ x + B @ delta_u_phys

        # --- reward ------------------------------------------------------
        err_metric = np.sum(err_scaled**2)
        proximity  = -err_metric + np.exp(-err_metric / SIGMA**2)
        progress   = 0.0 if prev_err_metric is None else (prev_err_metric - err_metric)
        reward     = 10*proximity + progress
        prev_err_metric = err_metric

        # --- log (scaled) ------------------------------------------------
        records.append({
            "run": run, "step": k,
            "Y1": y_scaled[0], "Y2": y_scaled[1], "Y3": y_scaled[2],
            "U1": u_opt[0], "U2": u_opt[1], "U3": u_opt[2], "U4": u_opt[3],
            "e1": err_scaled[0], "e2": err_scaled[1], "e3": err_scaled[2],
            "err_metric": err_metric,
            "int1": int_term[0], "int2": int_term[1],
            "int3": int_term[2], "int4": int_term[3],
            "reward": reward
        })

# ── 6. Save + plot (scaled values only) ───────────────────────────────────
df = pd.DataFrame.from_records(records)
df.to_csv(CSV_OUT, index=False)
print("✔  Log saved →", CSV_OUT)

fig, axs = plt.subplots(2, 4, figsize=(14,7), sharex=True, dpi=250)
axs = axs.flatten()
for run in range(N_RUNS):
    sub = df[df.run == run]
    axs[0].plot(sub.step, sub.Y1, alpha=0.4)
    axs[1].plot(sub.step, sub.Y2, alpha=0.4)
    axs[2].plot(sub.step, sub.Y3, alpha=0.4)
    axs[3].plot(sub.step, sub.reward, alpha=0.4)
    axs[4].plot(sub.step, sub.U1, alpha=0.4)
    axs[5].plot(sub.step, sub.U2, alpha=0.4)
    axs[6].plot(sub.step, sub.U3, alpha=0.4)
    axs[7].plot(sub.step, sub.U4, alpha=0.4)

titles = ["Y1 (scaled)", "Y2 (scaled)", "Y3 (scaled)", "Reward",
          "U1 (‑1…1)",  "U2 (‑1…1)",  "U3 (‑1…1)",  "U4 (‑1…1)"]
for ax, t in zip(axs, titles): ax.set_title(t); ax.grid(True)
for ax in axs[4:]: ax.set_xlabel("Step")
fig.suptitle("Optimised control — scaled CVs & MVs")
plt.tight_layout(); plt.show()
