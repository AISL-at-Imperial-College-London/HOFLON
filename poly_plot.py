# best_vs_median_plot.py
# ---------------------------------------------------------------
import pandas as pd, numpy as np, matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# ---------------------------------------------------------------
# 0. Input CSV files (strategy → path)
# ---------------------------------------------------------------
files = {
    "Historical - median"     : "multi_run_dataset.csv",
    "HOFLON - median" : "opt_control_dataset_with_reward.csv",
#    "AWR"          : "awr_run_dataset.csv",
    "IQL - median"          : "iql_run_dataset.csv",
}
missing = [p for p in files.values() if not Path(p).is_file()]
if missing:
    raise FileNotFoundError("Missing: " + ", ".join(missing))

# ---------------------------------------------------------------
# 1. Physical‑unit helpers  (shared across datasets)
# ---------------------------------------------------------------
POLY_SCALE  = 100.0
TEMP_SCALE  = 350.0
MONO_SCALE  = 100.0
F_I_SCALE   = 1.0          # already physical
def Tc_phys(u_Tc): return u_Tc * 120 + 280   # K

# ---------------------------------------------------------------
# 2. Determine best & median run for each strategy
# ---------------------------------------------------------------
selected = {}   # strat → {"best": id, "median": id, "df": DataFrame}

for name, path in files.items():
    df = pd.read_csv(path)
    run_reward = df.groupby("run_id")["reward"].sum().sort_values()
    best_id   = run_reward.idxmax()
    median_id = run_reward.index[len(run_reward)//2]

    # keep a copy with physical columns added
    df["time_h"]     = df["time_min"] / 60.0
    df["Poly_kgm3"]  = df["P"]   * POLY_SCALE
    df["Temp_K"]     = df["T"] * TEMP_SCALE
    df["Mono_kgm3"]  = df["A"]   * MONO_SCALE
    df["f_I_kgh"]    = df["u_I"] * F_I_SCALE
    df["T_c_K"]      = Tc_phys(df["u_Tc"])

    selected[name] = {"best": best_id, "median": median_id, "df": df}

# ---------------------------------------------------------------
# 3. Plot helper (returns axes list)
# ---------------------------------------------------------------
def make_figure(title):
    fig = plt.figure(figsize=(12, 6), dpi=250)
    gs  = gridspec.GridSpec(2, 3, figure=fig)
    ax_poly = fig.add_subplot(gs[0, 0])
    ax_temp = fig.add_subplot(gs[0, 1])
    ax_rew  = fig.add_subplot(gs[0, 2])
    ax_fI   = fig.add_subplot(gs[1, 0])
    ax_Tc   = fig.add_subplot(gs[1, 1])
    ax_mono = fig.add_subplot(gs[1, 2])

    for ax in (ax_poly, ax_mono): ax.set_ylabel("kg m$^{-3}$")
    ax_temp.set_ylabel("K")
    ax_Tc.set_ylabel("K")
    ax_fI.set_ylabel("kg h$^{-1}$")
    ax_rew.set_ylabel("Reward")
    for ax in (ax_poly, ax_temp, ax_rew, ax_mono, ax_fI, ax_Tc):
        ax.set_xlabel("time [h]"); ax.grid(alpha=0.4)

    ax_poly.set_ylim(0,   150);   ax_poly.axhline(100, ls="--", lw=1, c="k")
    ax_temp.set_ylim(345, 360)
    ax_rew.set_ylim(-150,  600)
    ax_fI.set_ylim(0.2,   1)
    ax_Tc.set_ylim(340, 350)
    ax_mono.set_ylim(390, 560)

    ax_poly.set_title("Polymer Conc.")
    ax_temp.set_title("Reactor Temp.")
    ax_mono.set_title("Monomer A Conc.")
    ax_fI.set_title("Initiator Feed")
    ax_Tc.set_title("Coolant Temp.")
    ax_rew.set_title("Reward")

    fig.suptitle(title, y=0.96)
    return fig, [ax_poly, ax_temp, ax_rew, ax_fI, ax_Tc, ax_mono]

# ---------------------------------------------------------------
# 4. Colour‑blind‑safe palette & style
# ---------------------------------------------------------------
palette = {
    "Historical - median":      "gray",
    "HOFLON - median":  "blue",
    "AWR":           "orange",
    "IQL - median":           "orange",
}
style_best   = dict(ls="-",  lw=1.4, alpha=0.85)
style_median = dict(ls="-", lw=1.4, alpha=0.85)

# ---------------------------------------------------------------
# 5. Plot BEST runs
# ---------------------------------------------------------------
fig_best, axes_best = make_figure(None)

for strat, info in selected.items():
    sub = info["df"][info["df"].run_id == info["best"]]
    c   = palette[strat]
    axes_best[0].plot(sub.time_h, sub.Poly_kgm3,  c=c, label=strat, **style_best)
    axes_best[1].plot(sub.time_h, sub.Temp_K,     c=c, **style_best)
    axes_best[2].plot(sub.time_h, sub.reward,     c=c, **style_best)
    axes_best[3].plot(sub.time_h, sub.f_I_kgh,    c=c, **style_best)
    axes_best[4].plot(sub.time_h, sub.T_c_K,      c=c, **style_best)
    axes_best[5].plot(sub.time_h, sub.Mono_kgm3,  c=c, **style_best)

fig_best.legend(loc="upper center", ncol=4, frameon=False)
plt.tight_layout(rect=[0,0.03,1,0.93])
plt.show()

# ---------------------------------------------------------------
# 6. Plot MEDIAN runs
# ---------------------------------------------------------------
fig_med, axes_med = make_figure(None)

for strat, info in selected.items():
    sub = info["df"][info["df"].run_id == info["median"]]
    c   = palette[strat]
    axes_med[0].plot(sub.time_h, sub.Poly_kgm3,  c=c, label=strat, **style_median)
    axes_med[1].plot(sub.time_h, sub.Temp_K,     c=c, **style_median)
    axes_med[2].plot(sub.time_h, sub.reward,     c=c, **style_median)
    axes_med[3].plot(sub.time_h, sub.f_I_kgh,    c=c, **style_median)
    axes_med[4].plot(sub.time_h, sub.T_c_K,      c=c, **style_median)
    axes_med[5].plot(sub.time_h, sub.Mono_kgm3,  c=c, **style_median)

fig_med.legend(loc="upper center", ncol=4, frameon=False)
plt.tight_layout(rect=[0,0.03,1,0.93])
plt.show()
