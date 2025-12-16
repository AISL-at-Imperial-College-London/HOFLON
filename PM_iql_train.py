# iql_fit_with_r2.py
# ----------------------------------------------------------------------
# 0) Imports
# ----------------------------------------------------------------------
import numpy as np, pandas as pd, joblib, matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor

# ----------------------------------------------------------------------
# 1) Load data
# ----------------------------------------------------------------------
df = pd.read_csv("paper_machine_batch_log_scaled.csv")
print("Data rows:", len(df))

x_cols      = ['Y1','Y2','Y3','e1','e2','e3','int1','int2','int3','int4']
action_cols = ['U1','U2','U3','U4']

# ----------------------------------------------------------------------
# 2) Train / hold‑out split (last 5 runs held out)
# ----------------------------------------------------------------------
runs       = sorted(df.run.unique())
test_runs  = runs[-5:]
train_runs = runs[:-5]

train_df = df[df.run.isin(train_runs)].reset_index(drop=True)
test_df  = df[df.run.isin(test_runs)].reset_index(drop=True)

print("Train runs :", train_runs[:3], "...", train_runs[-1])
print("Test runs  :", test_runs)

# ----------------------------------------------------------------------
# 3) Add next‑state indices & masks
# ----------------------------------------------------------------------
γ = 0.9

train_df['next_idx']  = train_df.index + 1
terminal_mask         = train_df.run != train_df.run.shift(-1)
train_df.loc[terminal_mask, 'next_idx'] = np.nan

X_s   = train_df[x_cols].values
R_s   = train_df.reward.values
next_mask = ~train_df.next_idx.isna().values
next_idx  = train_df.next_idx.dropna().astype(int).values

X_sn = np.empty_like(X_s); X_sn[:] = np.nan
X_sn[next_mask] = train_df.loc[next_idx, x_cols].values

# ----------------------------------------------------------------------
# 4) Expectile value function  Vτ(s)  (τ = 0.9)  + R² diagnostics
# ----------------------------------------------------------------------
τ = 0.85
V_model = XGBRegressor(
    objective="reg:squarederror",
    n_estimators=600,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
)

target = (train_df['reward']
          .iloc[::-1]
          .groupby(train_df.run)
          .cumsum()[::-1]
          .values)                 # Monte‑Carlo return

w = np.ones_like(target)
for it in range(3):
    V_model.fit(X_s, target, sample_weight=w)
    V_pred = V_model.predict(X_s)
    resid  = target - V_pred
    w      = np.where(resid >= 0, τ, 1-τ)

print(f"Vτ  train R²  : {r2_score(target, V_pred):8.4f}",
      f"   RMSE {mean_squared_error(target, V_pred):8.2f}")

joblib.dump(V_model, "iql_value_expectile.joblib")
print("✓ saved Vτ  → iql_value_expectile.joblib")

# ----------------------------------------------------------------------
# 5) Q‑function target:  r + γ Vτ(s')
# ----------------------------------------------------------------------
V_next           = np.zeros_like(R_s)
V_next[next_mask] = V_model.predict(X_sn[next_mask])
Q_target         = R_s + γ * V_next

Q_model = XGBRegressor(
    objective="reg:squarederror",
    n_estimators=600,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
).fit(
    np.hstack([X_s, train_df[action_cols].values]),
    Q_target
)

Q_pred = Q_model.predict(np.hstack([X_s, train_df[action_cols].values]))
print(f"Q̂  train R²  : {r2_score(Q_target, Q_pred):8.4f}",
      f"   RMSE {mean_squared_error(Q_target, Q_pred):8.2f}")

joblib.dump(Q_model, "iql_q_function.joblib")
print("✓ saved Q   → iql_q_function.joblib")

# ----------------------------------------------------------------------
# 6) Advantage and IQL weights
# ----------------------------------------------------------------------
A        = Q_pred - V_pred
β        = 1e-3
w_exp    = np.exp(np.clip(A / β, -10, 10))      # stable

# ----------------------------------------------------------------------
# 7) Policy training (IQL‑weighted) + R² per action dimension
# ----------------------------------------------------------------------
def fit_policy(X, Y, samp_w=None):
    models = []
    for k in range(Y.shape[1]):
        mdl = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=800,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        ).fit(X, Y[:, k], sample_weight=samp_w)
        models.append(mdl)
    return models

# X for policy = pure state features
iql_models = fit_policy(X_s, train_df[action_cols].values, samp_w=w_exp)
joblib.dump(iql_models, "iql_policy_weighted.joblib")
print("✓ saved IQL policy → iql_policy_weighted.joblib")

# ---------- R² for each action dimension (train set) -----------------
for k, mv in enumerate(action_cols):
    y_true = train_df[mv].values
    y_pred = iql_models[k].predict(X_s)
    print(f"{mv:>4s}  policy R²: {r2_score(y_true, y_pred):8.4f}   "
          f"RMSE {mean_squared_error(y_true, y_pred):8.3f}")
