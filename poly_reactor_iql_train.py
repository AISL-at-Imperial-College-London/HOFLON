# ----------------------------------------------------------------------
# 0) Imports (unchanged)
# ----------------------------------------------------------------------
import numpy as np, pandas as pd, joblib, matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from xgboost        import XGBRegressor

# ----------------------------------------------------------------------
# 1) Load data  (unchanged)
# ----------------------------------------------------------------------
csv_path = "multi_run_dataset.csv"
df = pd.read_csv(csv_path)
print("Loaded", csv_path, "→", df.shape[0], "rows")

# ----------------------------------------------------------------------
# 2) Feature / action columns (unchanged)
# ----------------------------------------------------------------------
x_cols = [
    "A", "I", "R",        # reactor species
    "P", "T",           # measurements
    "err_P", "err_T",     # tracking errors
    "int_err_P", "int_err_T"
]
action_cols = ["u_I", "u_Tc"]

# ----------------------------------------------------------------------
# 3) Hold‑out split (unchanged)
# ----------------------------------------------------------------------
all_runs      = sorted(df['run_id'].unique())
held_out_runs = all_runs[-5:]
train_runs    = all_runs[:-5]

print("Training on runs :", train_runs[:3], "...", train_runs[-1])
print("Held‑out runs    :", held_out_runs)

train_df = df[df['run_id'].isin(train_runs)].reset_index(drop=True)
test_df  = df[df['run_id'].isin(held_out_runs)].reset_index(drop=True)

# ----------------------------------------------------------------------
# 4)  Monte‑Carlo return  G_t  (undiscounted for simplicity)
# ----------------------------------------------------------------------
def add_cum_reward(d):
    d = d.sort_values(['run_id', 'time_min']).reset_index(drop=True)
    d['cum_reward'] = (
        d.iloc[::-1]
          .groupby('run_id')['reward']
          .cumsum()
          .iloc[::-1]
    )
    return d

train_df = add_cum_reward(train_df)
test_df  = add_cum_reward(test_df)

# ------------------------------------------------------------------ §5  (IQL)  Q–V fixed-point + AWR weights
γ       = 0.95
τ       = 0.90
β       = 12.0
w_clip  = 50.0
irwls_rounds = 2          # inner rounds for expectile IRLS
n_qv_iter    = 3          # outer V↔Q passes

# Ensure episode/time ordering and compute next-row indices per run
train_df = train_df.sort_values(['run_id','time_min']).reset_index(drop=True)
idx_next = (train_df.index + 1).to_numpy()
terminal = (train_df['run_id'] != train_df['run_id'].shift(-1))
idx_next[terminal.values] = -1
has_next = (idx_next != -1)

X_s = train_df[x_cols].values
A_s = train_df[action_cols].values
SA  = np.hstack([X_s, A_s])   # [state | action] features for Q(s,a)
R   = train_df['reward'].values

# Expectile weights helper
def _exp_weights(residual, tau):
    # residual = Q(s,a) - V(s)
    return np.where(residual >= 0.0, tau, 1.0 - tau)

# Models: one scalar Q over [s, a] (keeps your original shape), one scalar V over s
Q_model = XGBRegressor(
    objective="reg:squarederror",
    n_estimators=1200, learning_rate=0.03, max_depth=10,
    subsample=0.9, colsample_bytree=0.9, random_state=42, tree_method="hist"
)
V_model = XGBRegressor(
    objective="reg:squarederror",
    n_estimators=800, learning_rate=0.05, max_depth=8,
    subsample=0.9, colsample_bytree=0.9, random_state=42, tree_method="hist"
)

# Warm-start V with a crude fit to returns (optional; just helps first TD step)
V_model.fit(X_s, R)
V_pred = V_model.predict(X_s)

for it in range(n_qv_iter):
    # ---- Q step: fit to TD = r + γ V(s')
    td = R.copy()
    if has_next.any():
        td[has_next] += γ * V_pred[idx_next[has_next]]
    Q_model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=1200, learning_rate=0.03, max_depth=10,
        subsample=0.9, colsample_bytree=0.9, random_state=42, tree_method="hist"
    ).fit(SA, td)

    Q_pred = Q_model.predict(SA)

    # ---- V step: expectile regression Vτ(s) toward Q(s,a)
    V_model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=800, learning_rate=0.05, max_depth=8,
        subsample=0.9, colsample_bytree=0.9, random_state=42, tree_method="hist"
    ).fit(X_s, Q_pred)  # seed
    for _ in range(irwls_rounds):
        V_curr = V_model.predict(X_s)
        w_exp  = _exp_weights(Q_pred - V_curr, τ)
        V_model = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=800, learning_rate=0.05, max_depth=8,
            subsample=0.9, colsample_bytree=0.9, random_state=42, tree_method="hist"
        ).fit(X_s, Q_pred, sample_weight=w_exp)
    V_pred = V_model.predict(X_s)

    print(f"[IQL {it+1}/{n_qv_iter}] "
          f"V expectile RMSE vs Q: {np.sqrt(mean_squared_error(Q_pred, V_pred)):.4f}")

# Save the Q model (keeps your artifact name)
joblib.dump(Q_model, "xgb_qfunc.joblib")
print("✓ Q-function saved → xgb_qfunc.joblib")

# ------------------------------------------------------------------ §6  (IQL)  Final Q,V predictions
train_df['Q_pred'] = Q_pred
train_df['V_pred'] = V_pred

# ------------------------------------------------------------------ §7  (IQL)  Advantage & weights (no ε-floor)
train_df['advantage'] = train_df['Q_pred'] - train_df['V_pred']
w = np.exp(np.clip(train_df['advantage'] / β, a_min=-50, a_max=50))
train_df['weight'] = np.clip(w, 0.0, w_clip)

def fit_policy(X, Y, sample_weight=None):
    models = []
    for k, mv in enumerate(action_cols):
        mdl = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=600,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            tree_method="hist",
            random_state=42,
            n_jobs=-1,
        )
        if sample_weight is None:
            mdl.fit(X, Y[:, k])
        else:
            mdl.fit(X, Y[:, k], sample_weight=sample_weight)

        # quick train fit print (optional)
        y_hat = mdl.predict(X)
        print(f"π[{mv}]  R²={r2_score(Y[:,k], y_hat):.4f}  "
              f"RMSE={mean_squared_error(Y[:,k], y_hat)**0.5:.4f}")

        models.append(mdl)

    return {"models": models, "x_cols": x_cols, "action_cols": action_cols}

# ------------------------------------------------------------------ §9 (unchanged)  — fits BC and IQL policies with your same API
X_tr  = train_df[x_cols].values
Y_tr  = train_df[action_cols].values
w_tr  = train_df['weight'].values

bc_models  = fit_policy(X_tr, Y_tr)          # behaviour cloning
iql_models = fit_policy(X_tr, Y_tr, w_tr)    # IQL-weighted fit

joblib.dump(bc_models,  "xgb_policy_plain.joblib")
joblib.dump(iql_models, "xgb_policy_iql.joblib")
print("✓ plain BC  → xgb_policy_plain.joblib")
print("✓ IQL       → xgb_policy_iql.joblib")
