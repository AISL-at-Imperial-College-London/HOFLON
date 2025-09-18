# -*- coding: utf-8 -*-
"""
Optimisation-based control of a single-monomer polymerisation CSTR
(no dummy states, all logs in scaled units)

Author: ChatGPT – 27 Jun 2025
"""

from __future__ import annotations
import numpy as np, pandas as pd, torch, torch.nn as nn, joblib, xgboost
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import time, statistics          # make sure these are near your other imports
OPT_SOLVE_TIMES = []             # global list collects one entry per call
import torch, torch.nn as nn
import numpy as np

class TorchPredictWrapper:
    """Wraps a torch model & sklearn-style scaler, exposes .predict."""
    def __init__(self, torch_model: nn.Module, scaler, device="cpu"):
        self.model   = torch_model.to(device)
        self.scaler  = scaler
        self.device  = device

    def predict(self, arr: np.ndarray) -> np.ndarray:
        x = self.scaler.transform(arr).astype(np.float32)
        with torch.no_grad():
            return self.model(torch.tensor(x, device=self.device)).cpu().numpy()
# ── put these near the top of pcstr_scaled_mlp_contrl_online_deltauloss_QandAAE_var_w3.py ──
import torch, torch.nn as nn, numpy as np

class TinyQNet(nn.Module):
    """Must match the architecture used when the model was pickled."""
    def __init__(self, in_dim: int, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

class TorchPredictWrapper:
    """Same wrapper that was pickled; exposes .predict(arr)."""
    def __init__(self, torch_model: nn.Module, scaler, device="cpu"):
        self.model  = torch_model.to(device)
        self.scaler = scaler
        self.device = device
    def predict(self, arr: np.ndarray) -> np.ndarray:
        x = self.scaler.transform(arr).astype(np.float32)
        with torch.no_grad():
            return self.model(torch.tensor(x, device=self.device)).cpu().numpy()
# ─────────────────────────────────────────────────────────────────────────────

# ───────────────────────── 1. Physical & scaling constants ──────────────────
R_GAS = 8.314;  RHO = 1_000.0;  CP = 2_000.0
UA, V = 550.0, 1.0                           # J s⁻¹ K⁻¹, m³
W_A, W_S, T_FEED = 100.0, 80.0, 350.0        # kg h⁻¹, K
MW_A, MW_I = 0.100, 0.165                   # kg mol⁻¹
A_d,E_d, A_p,E_p, A_t,E_t = 1e9,125e3, 4e4,25e3, 1e6,15e3
DELTA_H = 100e3                             # J mol⁻¹ (exothermic +)

TC_MIN, TC_MAX = 280.0, 400.0               # K  (action 2)
I_MIN,  I_MAX  =   0.0,   2.5               # kg h⁻¹  (action 1)

S_CONC, S_TEMP = 100.0, 350.0               # scale divisors (conc, temp)

# ───────────────────────── 2. Five-state scaled ODE ─────────────────────────
def _k(A,E,T): return A*np.exp(-E/(R_GAS*T))

def cstr_ode_scaled(t, y5, u_I, Tc_s):
    """y5 = [A_s, I_s, R_s, P_s, T_s]  (all scaled)"""
    A_s,I_s,R_s,P_s,T_s = y5
    A,I,R,P,T = (*y5[:4]*S_CONC, T_s*S_TEMP)

    f_I = I_MIN  + u_I  * (I_MAX  - I_MIN)
    T_c = TC_MIN + Tc_s * (TC_MAX - TC_MIN)
    F   = (f_I + W_A + W_S)/(RHO*3600.0);  D = F/V

    k_d,k_p,k_t = _k(A_d,E_d,T),_k(A_p,E_p,T),_k(A_t,E_t,T)
    r_d=k_d*I/MW_I;         r_p=k_p*R*A/MW_A;     r_t=k_t*R**2

    dA=(W_A/3600)/V - D*A - r_p*MW_A
    dI=(f_I/3600)/V - D*I - r_d*MW_I
    dR= 2*r_d - 2*r_t
    dP= r_p*MW_A - D*P

    Q_rxn=DELTA_H*r_p*V;   Q_cool=UA*(T-T_c);  Q_flow=RHO*CP*F*(T_FEED-T)
    dT=(Q_rxn-Q_cool+Q_flow)/(RHO*CP*V)

    return [dA/S_CONC, dI/S_CONC, dR/S_CONC, dP/S_CONC, dT/S_TEMP]

# ─────────────────────── 3.  AAE & XGB models (9/11 dims) ───────────────────
class Encoder(nn.Module):
    def __init__(self, inp=9, lat=4):
        super().__init__(); self.net=nn.Sequential(
            nn.Linear(inp,64),nn.Tanh(),nn.Linear(64,16),nn.Tanh(),nn.Linear(16,lat))
    def forward(self,x): return self.net(x)

class Decoder(nn.Module):
    def __init__(self, out=9, lat=4):
        super().__init__(); self.net=nn.Sequential(
            nn.Linear(lat,16),nn.Tanh(),nn.Linear(16,64),nn.Tanh(),nn.Linear(64,out))
    def forward(self,z): return self.net(z)

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
enc,dec=Encoder().to(device),Decoder().to(device)
ckpt=torch.load("aae_weights.pth",map_location=device)
enc.load_state_dict(ckpt["encoder"]); dec.load_state_dict(ckpt["decoder"])
enc.eval(); dec.eval()
AE_SCALER=joblib.load("ae_scaler.pkl")
V_model = joblib.load("q_discounted.joblib")

# ───────────────────────── 4. Objective & weight switcher ───────────────────
def objective_core(u_I,Tc_s,
                   A_s,I_s,R_s,P_s,T_s,
                   int_P,int_T,u_prev,setpt,f1,f2,f3):
    err_P, err_T = setpt["P"]-P_s, setpt["T_s"]-T_s

    ae_in=np.array([A_s,I_s,R_s,P_s,T_s,err_P,err_T,u_I,Tc_s]).reshape(1,-1)
    ae_scl=AE_SCALER.transform(ae_in)
    with torch.no_grad():
        recon=dec(enc(torch.from_numpy(ae_scl).float().to(device))).cpu().numpy()
        recon_orig = AE_SCALER.inverse_transform(recon)
        recon_mse  = float(np.mean((ae_in - recon_orig) ** 2))

    int_P_s = int_P / 1000.0      #  scaled integrators
    int_T_s = int_T / 1000.0
    xgb_in=np.array([A_s,I_s,R_s,P_s,T_s,err_P,err_T,int_P_s,int_T_s,u_I,Tc_s]).reshape(1,-1)
    value_pred=float(V_model.predict(xgb_in)[0])

    du_I,du_Tc=u_I-u_prev[0],Tc_s-u_prev[1]
    return f1*recon_mse + f2*(-value_pred) + f3*(du_I**2+du_Tc**2)

def pick_weights(err_P,err_T,tol=0.04,n_consec=4):
    if not hasattr(pick_weights,"ctr"): pick_weights.ctr=0; pick_weights.sw=False
    pick_weights.ctr = pick_weights.ctr+1 if (abs(err_P)<tol and abs(err_T)<tol) else 0
    if pick_weights.ctr>=n_consec:
        if not pick_weights.sw: print(f"✓ corridor {n_consec} steps — weights→2,0.009,10"); pick_weights.sw=True
        return 0.5,6,0.025
    pick_weights.sw=False; return 0.5,6,0.025
    
# --------------------------------------------------------------------------
# dual-solver optimiser: Nelder-Mead vs Powell → choose lower objective
# --------------------------------------------------------------------------
import numpy as np
from scipy.optimize import minimize
try:                                    # DIRECT is in SciPy ≥ 1.11
    from scipy.optimize import direct
    HAVE_DIRECT = True
except ImportError:
    HAVE_DIRECT = False                 # graceful fallback
    # If DIRECT is missing you can: pip install --upgrade scipy>=1.11


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


# ────────────────────────────────────────────────────────────────────────────
def optim_controller(y_s, int_P, int_T, u_prev, setpt, dt):
    """
    MPC optimiser using three derivative-free solvers:
    1. Nelder-Mead   (local, no bounds)
    2. Powell        (local, box bounds)
    3. DIRECT        (global pattern search, box bounds)
    Returns: u_I, Tc_s, updated int_P, int_T
    """
    # ---------- 0. unpack & helpers ----------------------------------------
    A_s, I_s, R_s, P_s, T_s = y_s
    err_P = setpt["P"]   - P_s
    err_T = setpt["T_s"] - T_s
    f1, f2, f3 = pick_weights(err_P, err_T)

    def obj(u_vec):
        return objective_core(
            u_I=u_vec[0], Tc_s=u_vec[1],
            A_s=A_s, I_s=I_s, R_s=R_s, P_s=P_s, T_s=T_s,
            int_P=int_P, int_T=int_T,
            u_prev=u_prev, setpt=setpt,
            f1=f1, f2=f2, f3=f3,
        )

    bounds = [(0.0, 1.0), (0.0, 1.0)]   # shared hard box
    x0     = u_prev if np.any(u_prev) else np.array([0.7, 0.5])

    # ---------- 1. Nelder-Mead ---------------------------------------------
    res_nm = minimize(obj, x0, method="Nelder-Mead",
                      options={"maxiter": 250, "disp": False})
    u_nm_raw = res_nm.x
    u_nm     = np.clip(u_nm_raw, 0.0, 1.0)
    obj_nm   = obj(u_nm)

    # ---------- 2. Powell (bounded) ----------------------------------------
    res_pw = minimize(obj, x0, method="Powell",
                      bounds=bounds,
                      options={"maxiter": 250, "disp": False})
    u_pw_raw = res_pw.x
    u_pw     = np.clip(u_pw_raw, 0.0, 1.0)
    obj_pw   = obj(u_pw)

    # ---------- 3. DIRECT (pattern search) ---------------------------------
    if HAVE_DIRECT:
        res_ps = direct(obj, bounds, maxiter=40, eps=1e-1)   # DIRECT options
        u_ps_raw = res_ps.x
        u_ps     = np.clip(u_ps_raw, 0.0, 1.0)
        obj_ps   = obj(u_ps)
    else:
        # If DIRECT is unavailable, set cost huge so it never wins
        u_ps_raw, u_ps, obj_ps = (None, None, np.inf)

    # ---------- choose best of the three -----------------------------------
    objs = np.array([obj_nm, obj_pw, obj_ps])
    idx  = np.argmin(objs)

    if idx == 0:          # Nelder-Mead wins
        u_raw, u_vec = u_nm_raw, u_nm
    elif idx == 1:        # Powell wins
        u_raw, u_vec = u_pw_raw, u_pw
    else:                 # DIRECT wins
        u_raw, u_vec = u_ps_raw, u_ps

    # ---------- anti-wind-up (compare raw vs clipped) ----------------------
    if u_vec[0] == u_raw[0]:
        int_P += err_P * dt
    if u_vec[1] == u_raw[1]:
        int_T += err_T * dt

    return float(u_vec[0]), float(u_vec[1]), int_P, int_T

# ───────────────────────── 6. Simulation driver (multi-run) ─────────────────
# --------------------------------------------------------------------------
# 0.  scaled measurement-noise std-devs (physical σ → scaled σ)
# --------------------------------------------------------------------------
SIG_MONO_S = 2.0 / S_CONC          # 2 kg m⁻³  ÷ 100  = 0.02
SIG_POLY_S = 0.5 / S_CONC          # 0.5 kg m⁻³ ÷ 100  = 0.005
SIG_TEMP_S = 0.2 / S_TEMP          # 0.2 K     ÷ 350  ≈ 5.714e-4
# --------------------------------------------------------------------------
# 1.  multi-run simulation driver (noise added to measurements + logs)
# --------------------------------------------------------------------------
def simulate_many_runs(n_runs=100,
                       t_total=100*3600,
                       dt=30*60,
                       seed=0):
    """
    Returns
    -------
    df_logs   : full trajectory data (one row per Δt)
    df_times  : timing stats (one row per run)
    """
    rng   = np.random.default_rng(seed)
    setpt = {"P": 1.0, "T_s": 1.0}         # scaled set-points

    logs, time_summaries = [], []          # <── two collectors

    # fixed scaled initial state --------------------------------------------
    y0 = np.array([5.55555456,   # A_s
                   0.0,          # I_s
                   0.0,          # R_s
                   0.0,          # P_s
                   1.0])         # T_s

    for run in range(n_runs):
        print(f"▶ run {run+1}/{n_runs}")

        # ── per-run initialisation ─────────────────────────────────────────
        OPT_SOLVE_TIMES = []              # fresh list for this run

        y_s      = y0.copy()
        int_P    = int_T = 0.0
        u_prev   = np.zeros(2)
        t        = 0.0
        n_steps  = int(t_total // dt)
        
        prev_poly_meas = None
        prev_Ts_meas   = None
        prev_u         = np.zeros(2)   # same shape as u_prev

        for _ in range(n_steps):

            # 1. add measurement noise -------------------------------------
            y_meas = y_s.copy()
            y_meas[0] += rng.normal(0.0, SIG_MONO_S)   # monomer
            y_meas[3] += rng.normal(0.0, SIG_POLY_S)   # polymer
            y_meas[4] += rng.normal(0.0, SIG_TEMP_S)   # temperature

             # 2) controller uses noisy measurements
            t0 = time.perf_counter()
            u_I, Tc_s, int_P, int_T = optim_controller(y_meas, int_P, int_T, u_prev, setpt, dt)
            OPT_SOLVE_TIMES.append(time.perf_counter() - t0)
            
            # 3) integrate true plant (noise-free)
            sol = solve_ivp(
                lambda tt, yy: cstr_ode_scaled(tt, yy, u_I, Tc_s),
                [0, dt], y_s,
                method="BDF", atol=1e-8, rtol=1e-6
            )
            y_s = sol.y[:, -1]
            t  += dt
            
            # --- reward on noisy states -----------------------------------
            if (prev_poly_meas is None) or (prev_Ts_meas is None):
                reward = 0.0   # no previous sample yet
            else:
                reward = compute_reward(
                    poly_t   = y_meas[3],          # current noisy polymer
                    T_s_t    = y_meas[4],          # current noisy temperature
                    poly_prev= prev_poly_meas,     # previous noisy polymer
                    T_s_prev = prev_Ts_meas,       # previous noisy temperature
                    u_t      = np.array([u_I, Tc_s]),
                    u_prev   = prev_u,             # previous action
                    setpts   = setpt,              # <-- FIX: pass the dict, not y_s
                    λ1       = 1.0,
                    λ2       = 1.0
                )
            
            # update "previous" trackers for next step
            prev_poly_meas = y_meas[3]
            prev_Ts_meas   = y_meas[4]
            prev_u         = np.array([u_I, Tc_s])
            
            # now update u_prev used by controller anti-windup in next call
            u_prev = np.array([u_I, Tc_s])
            # 4. tracking errors (noisy) -----------------------------------
            err_P_s = setpt["P"]   - y_meas[3]
            err_T_s = setpt["T_s"] - y_meas[4]

            # 5. logging ---------------------------------------------------
            logs.append(dict(
                run_id     = run,
                time_min   = t / 60,
                A          = y_meas[0],
                I          = y_meas[1],
                R_rad      = y_meas[2],
                P          = y_meas[3],
                T        = y_meas[4],
                err_P    = err_P_s,
                err_T    = err_T_s,
                int_err_P  = int_P,
                int_err_T  = int_T,
                u_I        = u_I,
                u_Tc       = Tc_s,
                reward     = reward
            ))

        # ── per-run timing summary (after inner loop) ----------------------
        if OPT_SOLVE_TIMES:                 # avoid zero-division
            n_calls = len(OPT_SOLVE_TIMES)
            mean_ms = statistics.mean(OPT_SOLVE_TIMES)  * 1e3
            med_ms  = statistics.median(OPT_SOLVE_TIMES)* 1e3
            max_ms  = max(OPT_SOLVE_TIMES)              * 1e3
            print(f"   optimisation time: {n_calls} calls — "
                  f"mean {mean_ms:.1f} ms, median {med_ms:.1f} ms, "
                  f"max {max_ms:.1f} ms")

            time_summaries.append(dict(
                run_id    = run,
                n_calls   = n_calls,
                mean_ms   = mean_ms,
                median_ms = med_ms,
                max_ms    = max_ms
            ))

    # 6. return both tables --------------------------------------------------
    return pd.DataFrame(logs), pd.DataFrame(time_summaries)

# ───────────────────────── 7. Run & quick plots ─────────────────────────────
if __name__ == "__main__":
    df, dft = simulate_many_runs(n_runs=1, seed=11)


    print("\n=== optimiser timing per run (ms) ===")
    print(dft.round(1))

    df.to_csv("opt_control_dataset_with_reward.csv",index=False)

    # physical for plots
    df["P_phys"]=df["P"]*S_CONC; df["T_phys"]=df["T"]*S_TEMP
    df["f_I"]=I_MIN+df["u_I"]*(I_MAX-I_MIN)
    df["T_c"]=TC_MIN+df["u_Tc"]*(TC_MAX-TC_MIN)

    fig,axs=plt.subplots(2,2,figsize=(14,9),dpi=250)
    for _,g in df.groupby("run_id"):
        axs[0,0].plot(g.time_min,g.P_phys,alpha=.3)
        axs[0,1].plot(g.time_min,g.T_phys,alpha=.3)
        axs[1,0].plot(g.time_min,g.f_I,   alpha=.3)
        axs[1,1].plot(g.time_min,g.T_c,   alpha=.3)
    axs[0,0].set(title="Polymer",ylabel="kg m$^{-3}$")
    axs[0,1].set(title="Temperature",ylabel="K")
    axs[1,0].set(title="Initiator feed",ylabel="kg h$^{-1}$")
    axs[1,1].set(title="Coolant T",ylabel="K")
    for ax in axs.flat: ax.set_xlabel("Time (min)"); ax.grid(True)
    plt.tight_layout(); plt.show()
