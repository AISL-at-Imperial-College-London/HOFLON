# ============================================================================
# 0) Imports
# ============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib                         # ← NEW

# ============================================================================
# 1) Global constants
# ============================================================================
N_RUNS   = 10
N_STEPS  = 250
SIGMA    = 0.2

# ------------------------------------------------------------------ CV scaling
Y_nom = np.array([64.0, 7.0, 9.0])
def scale_cv(y_phys):         return (y_phys - Y_nom) / Y_nom
def descale_cv(y_scaled):     return Y_nom * (1.0 + y_scaled)

# ------------------------------------------------------------------ MV scaling
u_min = np.array([300., 0., 1., 800.])
u_max = np.array([600., 200., 3., 1000.])
U_nom = np.array([450., 100., 2., 950.])

u_range = u_max - u_min
def scale_mv(u_phys):         return  2*(u_phys - u_min)/u_range - 1.0     # [-1,1]
def descale_mv(u_scaled):     return u_min + 0.5*u_range*(u_scaled + 1.0)

u_scaled_nom = scale_mv(U_nom)

# ------------------------------------------------------------------ set‑point
Ysp_phys   = np.array([102.0, 17.0, 9.0])
Ysp_scaled = scale_cv(Ysp_phys)

# ============================================================================
# 2)  State‑space matrices  (keep your A, B, C as before)
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

# ============================================================================
# 3)  Load behaviour‑cloning policy (UNweighted by default)
#     ------------------------------------------------------
#     • swap file name to 'xgb_policy_weighted.joblib' if you prefer
# ============================================================================
bc_models = joblib.load("iql_policy_weighted.joblib")      # list of 4 XGB models

x_cols = ['Y1','Y2','Y3','e1','e2','e3','int1','int2','int3','int4']
action_cols = ['U1','U2','U3','U4']

def policy_action(feature_vec):
    """
    Parameters
    ----------
    feature_vec : (10,) ndarray in the SAME order as `x_cols`

    Returns
    -------
    u_scaled : (4,) ndarray in [-1,1]^4  (already clipped)
    """
    preds = [m.predict(feature_vec.reshape(1,-1))[0] for m in bc_models]
    return np.clip(np.array(preds, dtype=np.float32), -1.0, 1.0)

# ============================================================================
# 4)  Simulation
# ============================================================================
records = []

for run in range(N_RUNS):
    print(f"Run {run+1}/{N_RUNS}")
    x = np.random.uniform(-0.1, 0.1, size=10)   # state deviation

    int_term       = np.zeros(4)                # four integrators
    u_scaled       = u_scaled_nom.copy()
    prev_err_metric = None

    for k in range(N_STEPS):
        # ------------------------------------------------------------------
        # 1) Plant measurement  (add ±3 % noise, as in log)
        # ------------------------------------------------------------------
        y_dev_phys = C @ x
        y_phys     = (Y_nom + y_dev_phys) * np.random.uniform(0.97, 1.03, 3)
        y_scaled   = scale_cv(y_phys)

        # ------------------------------------------------------------------
        # 2)  Behaviour‑cloning policy  (replaces PI block)
        # ------------------------------------------------------------------
        err_scaled = Ysp_scaled - y_scaled

        # --- integrator update (same rule as in original log)
        int_term += np.array([err_scaled[0], err_scaled[1],
                              err_scaled[2], err_scaled[0]])

        # --- build 10‑element feature vector  (MUST match x_cols order)
        feat_vec = np.array([
            y_scaled[0], y_scaled[1], y_scaled[2],        # Y1‑Y3
            err_scaled[0], err_scaled[1], err_scaled[2],  # e1‑e3
            int_term[0],  int_term[1], int_term[2], int_term[3]
        ], dtype=np.float32)

        # --- query policy
        u_scaled = policy_action(feat_vec)

        # --- anti‑wind‑up: freeze integrator if saturated & pushing out
        for i in range(4):
            e_i = err_scaled[0] if i in (0,3) else err_scaled[i-1]
            sat_lo = (u_scaled[i] <= -1.0) and (e_i < 0)
            sat_hi = (u_scaled[i] >=  1.0) and (e_i > 0)
            if sat_lo or sat_hi:
                int_term[i] -= e_i      # undo last add

        # ------------------------------------------------------------------
        # 3)  Plant state update  (needs physical Δu)
        # ------------------------------------------------------------------
        delta_u_phys = descale_mv(u_scaled) - U_nom
        x = A @ x + B @ delta_u_phys

        # ------------------------------------------------------------------
        # 4)  Reward (unchanged)
        # ------------------------------------------------------------------
        err_metric = np.sum(err_scaled**2)
        proximity  = -err_metric + np.exp(-err_metric / (SIGMA**2))
        progress   = 0.0 if prev_err_metric is None else (prev_err_metric - err_metric)
        reward     = 10*proximity + progress
        prev_err_metric = err_metric

        # ------------------------------------------------------------------
        # 5)  Log step
        # ------------------------------------------------------------------
        records.append({
            "run": run, "step": k,
            "Y1": y_scaled[0], "Y2": y_scaled[1], "Y3": y_scaled[2],
            "U1": u_scaled[0], "U2": u_scaled[1], "U3": u_scaled[2], "U4": u_scaled[3],
            "e1": err_scaled[0], "e2": err_scaled[1], "e3": err_scaled[2],
            "err_metric": err_metric,
            "int1": int_term[0], "int2": int_term[1],
            "int3": int_term[2], "int4": int_term[3],
            "reward": reward
        })

# ============================================================================
# 5)  Save & plot (unchanged)
# ============================================================================
df = pd.DataFrame.from_records(records)
df.to_csv("iql_policy_weighted.csv", index=False)
print("✔  Log saved → paper_machine_batch_log_iql.csv")

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
fig.suptitle("All runs — scaled CVs, reward, and scaled MVs (‑1…1)")
plt.tight_layout(); plt.show()
