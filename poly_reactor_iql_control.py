# -*- coding: utf-8 -*-
"""
poly_cstr_scaled_pid.py
=======================
Single-monomer CSTR with scaled states (conc/100, T/350) and 0-to-1 inputs,
driven by a 2-loop PI controller.  The solver holds each control move for a
15-minute interval (900 s) and logs only those sample points.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import joblib


# ============================================================================
# 3)  Load policy (weighted by advantage, 2 actions)
# ============================================================================
obj = joblib.load("xgb_policy_iql.joblib")
bc_models = obj if isinstance(obj, list) else obj["models"]

x_cols = [
    "A", "I", "R",           # concentrations
    "P", "T_s",              # measurements
    "err_P", "err_T",        # tracking errors
    "int_err_P", "int_err_T" # integrator states
]

action_cols = ["u_I", "u_Tc"]

# ── physical constants & geometry ─────────────────────────────────────────────
R, rho, Cp, UA, V = 8.314, 1_000.0, 2_000.0, 550.0, 1.0
W_A, W_s, T_feed = 100.0, 80.0, 350.0              # feeds
MW_A, MW_I = 0.100, 0.165
A_d, E_d, A_p, E_p, A_t, E_t = 1e9, 125e3, 4e4, 25e3, 1e6, 15e3
ΔH_rxn = 100e3                                      # J mol⁻¹

# ── scaling constants (inputs & states) ───────────────────────────────────────
TC_MIN, TC_MAX = 280.0, 400.0                      # K
I_MIN,  I_MAX  = 0.0,   2.5                        # kg h⁻¹
S_CONC, S_TEMP = 100.0, 350.0                      # state divisors

# ── Arrhenius helper ----------------------------------------------------------
def _k(A, E, T): return A * np.exp(-E / (R * T))

# ── scaled ODE system ---------------------------------------------------------
def reactor_odes_scaled(t, y_s, u_I_s, u_Tc_s):
    A, I, R_, P, T = (*y_s[:4] * S_CONC, y_s[4] * S_TEMP)
    f_I = I_MIN + u_I_s * (I_MAX - I_MIN)          # kg h⁻¹
    T_c = TC_MIN + u_Tc_s * (TC_MAX - TC_MIN)      # K

    F = (f_I + W_A + W_s) / (rho*3600)      # m³ s⁻¹, placeholder
    D = F / V

    k_d, k_p, k_t = _k(A_d, E_d, T), _k(A_p, E_p, T), _k(A_t, E_t, T)
    r_d = k_d * I / MW_I
    r_p = k_p * R_ * A / MW_A
    r_t = k_t * R_**2

    dA = (W_A/3600)/V - D*A - r_p*MW_A
    dI = (f_I/3600)/V - D*I - r_d*MW_I
    dR =  2*r_d - 2*r_t
    dP =  r_p*MW_A - D*P

    Q_rxn  = ΔH_rxn * r_p * V
    Q_cool = UA * (T - T_c)
    Q_flow = rho*Cp*F * (T_feed - T)
    dT     = (Q_rxn - Q_cool + Q_flow) / (rho*Cp*V)

    return [dA/S_CONC, dI/S_CONC, dR/S_CONC, dP/S_CONC, dT/S_TEMP]

def policy_action(feature_vec):
    """
    Parameters
    ----------
    feature_vec : ndarray of shape (10,)
        Ordered according to `x_cols`

    Returns
    -------
    u_scaled : ndarray of shape (2,)
        Scaled actions ∈ [−1, 1]² (already clipped)
    """
    preds = [m.predict(feature_vec.reshape(1, -1))[0] for m in bc_models]
    return np.clip(np.array(preds, dtype=np.float32), -1.0, 1.0)


# ── PI controller with anti-wind-up (in scaled space) ------------------------
def awr_control(x_state, setpts, dt):
    """
    AWR-based controller for polymerization CSTR with two control loops.

    Parameters
    ----------
    x_state : ndarray of shape (5,) — [A, I, R, P, T_s]
    setpts : dict with keys "P" and "T_s"
    dt : float — time step in seconds

    Returns
    -------
    u_I, u_Tc : float — clipped control inputs in [0, 1]
    int_P, int_T : float — updated integrator states
    """

    # --- Initialize integrator state on first call
    if not hasattr(awr_control, "int_P"):
        awr_control.int_P = 0.0
        awr_control.int_T = 0.0

    A, I, R, P_s, T_s = x_state
    err_P = setpts["P"] - P_s
    err_T = setpts["T_s"] - T_s

    # --- Update integrators
    awr_control.int_P += err_P * dt
    awr_control.int_T += err_T * dt

    # --- Feature vector (10 elements matching x_cols)
    feat_vec = np.array([
        A, I, R,
        P_s, T_s,
        err_P, err_T,
        awr_control.int_P,
        awr_control.int_T
    ], dtype=np.float32)

    # --- Get scaled actions from policy
    u_scaled = policy_action(feat_vec).flatten()

    # --- Clip to [0, 1] physical range
    u_I  = float(np.clip(u_scaled[0], 0.0, 1.0))
    u_Tc = float(np.clip(u_scaled[1], 0.0, 1.0))

    # --- Anti-windup (after clipping)
    if (u_I <= 0.0 and err_P < 0) or (u_I >= 1.0 and err_P > 0):
        awr_control.int_P -= err_P * dt
    if (u_Tc <= 0.0 and err_T < 0) or (u_Tc >= 1.0 and err_T > 0):
        awr_control.int_T -= err_T * dt

    return u_I, u_Tc, awr_control.int_P, awr_control.int_T



# ── hybrid reward (scaled states & inputs 0-1) ───────────────────────────────
def compute_reward(poly_t, T_s_t,              # state at k+1  (scaled)
                   poly_prev, T_s_prev,        # state at k
                   u_t, u_prev,                # actions (u_I, u_Tc) at k, k-1
                   setpts: dict[str, float],
                   λ1: float, λ2: float,       # action-smoothing weights
                   σ: float = 0.05) -> float:  # proximity width
    """
    + Proximity term   → large when error_now ≈ 0
    + Progress term    → positive when total MSE decreases
    − Control penalty  → discourage large Δu
    """
    # instantaneous squared error
    err_now  = (poly_t - setpts["P"])**2 + (T_s_t - setpts["T_s"])**2
    err_prev = (poly_prev - setpts["P"])**2 + (T_s_prev - setpts["T_s"])**2
    Δerr     = err_prev - err_now                       # improvement

    proximity = -err_now + np.exp(-err_now / σ**2)
    progress  = Δerr

    ΔI, ΔTc = u_t[0] - u_prev[0], u_t[1] - u_prev[1]
    ctrl_pen = λ1 * ΔI**2 + λ2 * ΔTc**2

    return 50*proximity + 5_000*progress - 0*ctrl_pen



# ── multi-run driver with measurement noise ────────────────────────────────
def simulate_many_runs(n_runs: int = 20,
                       t_total: float = 100 * 3600,
                       dt: float = 30 * 60,
                       setpts: dict[str, float] | None = None,
                       seed: int | None = None):

    rng = np.random.default_rng(seed)
    setpts = setpts or {"P": 1.0, "T_s": 1.0}

    # base PI gains (hand-tuned)
    base = dict(Kp_P=2.6*0.4, Ki_P=0.0002*0.4,
                Kp_T=0.03*0.4, Ki_T=0.000095*0.4)

    # scaled 1-σ noise (monomer ,- ,- ,polymer ,temperature)
    NOISE_STD = np.array([0.02, 0.0, 0.0, 0.005, 0.0005714])

    all_logs: list[pd.DataFrame] = []

    for run in range(n_runs):
        print(f"▶ run {run + 1}/{n_runs}")

        # deterministic initial (scaled) state
        y_s = np.array([5.55555456,
                        0,
                        0,
                        0,
                        1])

        int_P = int_T = 0.0
        t_clock = 0.0

        # randomise gains once per run
        mult = rng.uniform(0.3, 1.0, size=4)
        Kp_P, Ki_P, Kp_T, Ki_T = mult * np.fromiter(base.values(), float)

        # trackers for reward (use noisy values consistently)
        poly_prev = y_s[3]
        Ts_prev   = y_s[4]
        u_prev    = np.zeros(2)

        n_steps = int(t_total // dt)
        log: list[dict] = []

        for _ in range(n_steps):

            # --- measurement noise -----------------------------------------
            noise = rng.normal(0.0, NOISE_STD)
            y_meas = y_s + noise        # what the controller “sees”

            # --- AWR control on noisy measurements -------------------------
            A, I, R, P, T_s = y_meas
            x_state = np.array([A, I, R, P, T_s])
            u_I, u_Tc, int_P, int_T = awr_control(x_state, setpts, dt)

            # --- integrate true plant -------------------------------------
            sol = solve_ivp(
                lambda t, y: reactor_odes_scaled(t, y, u_I, u_Tc),
                [0, dt], y_s, method="BDF",
                atol=1e-8, rtol=1e-6
            )
            y_s = sol.y[:, -1]          # true next state (still scaled)
            t_clock += dt
            u_prev = np.array([u_I, u_Tc])

            # --- reward on noisy states -----------------------------------
            reward = compute_reward(
                y_meas[3], y_meas[4],          # current noisy
                poly_prev, Ts_prev,            # previous noisy
                np.array([u_I, u_Tc]), u_prev,
                setpts,
                λ1=1.0, λ2=1.0
            )
            poly_prev, Ts_prev = y_meas[3], y_meas[4]

            # --- log noisy measurements -----------------------------------
            A, I, R_, P, T_s = y_meas
            log.append({
                "run_id": run,
                "time_min": t_clock / 60,
                "A": A, "I": I, "R": R_,
                "P": P, "T": T_s,
                "err_P": setpts["P"] - P,
                "err_T": setpts["T_s"] - T_s,
                "int_err_P": int_P / 1000,
                "int_err_T": int_T / 1000,
                "u_I": u_I, "u_Tc": u_Tc,
                "reward": reward
            })

        all_logs.append(pd.DataFrame(log))

    return pd.concat(all_logs, ignore_index=True)


# ── script entry-point --------------------------------------------------------
if __name__ == "__main__":
    df = simulate_many_runs(n_runs=1,            # ← choose how many runs
                            t_total=100*3600,      # 24 h each (optional)
                            dt=30*60,             # 30-min sample
                            seed=123)
    df.to_csv("iql_run_dataset.csv", index=False)

# ---------------------------------------------------------------------------
#  convert scaled → physical just once
#  (adds four new columns: P_phys, T_phys, f_I_phys, T_c_phys)
# ---------------------------------------------------------------------------
df["P_phys"]   = df["P"]  * S_CONC
df["T_phys"]   = df["T"]  * S_TEMP
df["f_I_phys"] = I_MIN  + df["u_I"]  * (I_MAX  - I_MIN)
df["T_c_phys"] = TC_MIN + df["u_Tc"] * (TC_MAX - TC_MIN)


# ---------------------------------------------------------------------------
# overlay every run (semi-transparent)  – physical units
# ---------------------------------------------------------------------------
fig, axs = plt.subplots(2, 2, figsize=(14, 10), dpi=250)

for run_id, g in df.groupby("run_id"):
    axs[0, 0].plot(g["time_min"], g["P_phys"],   alpha=0.3)
    axs[0, 1].plot(g["time_min"], g["T_phys"],   alpha=0.3)
    axs[1, 0].plot(g["time_min"], g["f_I_phys"], alpha=0.3)
    axs[1, 1].plot(g["time_min"], g["T_c_phys"], alpha=0.3)

axs[0, 0].set(title="Polymer concentration", ylabel="P  [kg m$^{-3}$]")
axs[0, 1].set(title="Reactor temperature",   ylabel="T  [K]")
axs[1, 0].set(title="Initiator feed",        ylabel="f_I  [kg h$^{-1}$]")
axs[1, 1].set(title="Coolant temperature",   ylabel="T_c  [K]")

for ax in axs.flat:
    ax.set_xlabel("Time (min)")
    ax.grid(True)

plt.tight_layout()
plt.show()


# ---------------------------------------------------------------------------
# reward figure (one line per run)
# ---------------------------------------------------------------------------
plt.figure(figsize=(10, 5), dpi=250)
for run_id, g in df.groupby("run_id"):
    plt.plot(g["time_min"], g["reward"], alpha=0.4)

plt.title("Reward trajectories (all runs)")
plt.xlabel("Time (min)")
plt.ylabel("Reward")
plt.grid(True)
plt.tight_layout()
plt.show()