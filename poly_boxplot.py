# compare_strategies.py
# ---------------------------------------------------------------
# 0. Imports
# ---------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------------
# 1. CSV files  (edit paths / names as needed)
# ---------------------------------------------------------------
strategies = [
    ("Historical",   "multi_run_dataset.csv"),
    ("HOFLON RL", "opt_control_dataset_with_reward.csv"),
#    ("AWR",          "awr_run_dataset.csv"),
    ("IQL",          "iql_run_dataset.csv"),
]

# sanity‑check files
for name, f in strategies:
    if not Path(f).is_file():
        raise FileNotFoundError(f"Cannot find {f}")

# ---------------------------------------------------------------
# 2. Helper: per‑run RMSE for err_P, err_T
# ---------------------------------------------------------------
error_cols = ["err_P", "err_T"]   # must exist in every file

def per_run_rmse(df):
    return (
        df.groupby("run_id")[error_cols]
          .apply(lambda g: np.sqrt((g**2).mean()))
          .reset_index()
    )

# ---------------------------------------------------------------
# 3. Load everything & compute metrics
# ---------------------------------------------------------------
rmse_frames   = []
reward_series = []

for name, csv in strategies:
    df = pd.read_csv(csv)
    rmse = per_run_rmse(df)
    rmse["strategy"] = name
    rmse_frames.append(rmse)

    cum_reward = df.groupby("run_id")["reward"].sum()
    reward_series.append(pd.Series(cum_reward.values,
                                   name=name))

rmse_all   = pd.concat(rmse_frames, ignore_index=True)
reward_all = pd.concat(reward_series, axis=1)

# ---------------------------------------------------------------
# 4. Plot tracking RMSE  (independent y‑axes for P and T)
# ---------------------------------------------------------------
labels_xy = ["P (err_P)", "T (err_T)"]
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False, dpi=120)

for j, col in enumerate(error_cols):
    data = [
        rmse_all.loc[rmse_all["strategy"] == name, col].values
        for name, _ in strategies
    ]
    axes[j].boxplot(data, labels=[n for n, _ in strategies], showmeans=True)
    axes[j].set_title(f"Tracking RMSE – {labels_xy[j]}")
    axes[j].set_xlabel("Strategy")
    axes[j].set_ylabel("RMSE")           # each axis has its own scale

plt.tight_layout()
plt.show()

# ---------------------------------------------------------------
# 5. Plot cumulative reward per run
# ---------------------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(6, 4), dpi=250)
ax2.boxplot(
    [reward_all[name].dropna().values for name, _ in strategies],
    labels=[n for n, _ in strategies], showmeans=True
)
ax2.set_title("Cumulative Reward per Run")
ax2.set_ylabel("Σ reward")
plt.tight_layout()
plt.show()
