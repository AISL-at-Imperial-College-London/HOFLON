import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ----------------------------------------------------------------------
# 0. CSV files for the four strategies
# ----------------------------------------------------------------------
files = {
    "Historical"     : "paper_machine_batch_log_scaled.csv",
    "HOFLON RL" : "paper_machine_opt_log_scaled.csv",
    "IQL"          : "iql_policy_weighted.csv",
}

# ensure all files exist
missing = [f for f in files.values() if not Path(f).is_file()]
if missing:
    raise FileNotFoundError("Missing files:\n  " + "\n  ".join(missing))

# ----------------------------------------------------------------------
# 1. Settings
# ----------------------------------------------------------------------
error_cols = ["e1", "e2", "e3"]       # columns containing the tracking errors
labels_xy  = ["Y1 basis weight\n(e1)", "Y2 ash\n(e2)", "Y3 moisture\n(e3)"]

# ----------------------------------------------------------------------
# 2. Helper: per‑run RMSE
# ----------------------------------------------------------------------
def per_run_rmse(df: pd.DataFrame) -> pd.DataFrame:
    return (df.groupby("run")[error_cols]
              .apply(lambda g: np.sqrt((g**2).mean()))
              .reset_index())

# ----------------------------------------------------------------------
# 3. Load each log and compute metrics
# ----------------------------------------------------------------------
rmse_by_strat   = {}
reward_by_strat = {}

for name, path in files.items():
    df = pd.read_csv(path)
    rmse_by_strat[name]   = per_run_rmse(df)
    reward_by_strat[name] = df.groupby("run")["reward"].sum()

# ----------------------------------------------------------------------
# 4. Plot tracking RMSE (one subplot per variable)
# ----------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False, dpi=250)

for j, col in enumerate(error_cols):
    # gather data across strategies for this error column
    box_data = [rmse_by_strat[name][col].values for name in files.keys()]
    axes[j].boxplot(box_data, labels=list(files.keys()), showmeans=True)
    axes[j].set_title(f"Tracking RMSE – {labels_xy[j]}")
    axes[j].set_xlabel("Strategy")
    if j == 0:
        axes[j].set_ylabel("RMSE")

plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------
# 5. Plot cumulative reward per run
# ----------------------------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(6, 4), dpi=250)
box_reward = [reward_by_strat[name].values for name in files.keys()]
ax2.boxplot(box_reward, labels=list(files.keys()), showmeans=True)
ax2.set_title("Cumulative Reward per Run")
ax2.set_ylabel("Σ reward")
plt.tight_layout()
plt.show()
