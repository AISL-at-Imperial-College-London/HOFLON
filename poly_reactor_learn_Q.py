# =============================================================================
# 0) Imports
# =============================================================================
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from xgboost               import XGBRegressor
import joblib

CSV_PATH   = "multi_run_dataset.csv"       #  ← adjust if needed
MODEL_FILE = "q_discounted.joblib"
GAMMA      = 0.90                                #  discount factor (0.97–0.995)

# =============================================================================
# 1) Load log
# =============================================================================
df = pd.read_csv(CSV_PATH)


# =============================================================================
# 2) Discounted return  G_t = Σ_k γᵏ r_{t+k}
# =============================================================================
df = df.sort_values(['run_id', 'time_min']).reset_index(drop=True)

disc_return = np.zeros(len(df))
for run_id, grp in df.groupby('run_id', sort=False):
    g = 0.0
    for idx in reversed(grp.index):
        g = df.at[idx, 'reward'] + GAMMA * g
        disc_return[idx] = g
df['disc_return'] = disc_return

# =============================================================================
# 3) Features   s_t  +  a_t
# =============================================================================
state_cols = [
    "A", "I", "R",           # scaled concentrations
    "P", "T",                # polymer & temperature (scaled)
    "err_P", "err_T",        # scaled set-point errors
    "int_err_P", "int_err_T" # integrator states  (already ÷1000 in log)
]
action_cols = ['u_I','u_Tc']
sa_cols     = state_cols + action_cols

X = df[sa_cols].values.astype(np.float32)
y = df['disc_return'].values.astype(np.float32)

X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# =============================================================================
# 4) XGBoost Q-function
#    (fixed #trees for universal compatibility; raise n_estimators if needed)
# =============================================================================
Q_model = XGBRegressor(
    objective        = "reg:squarederror",
    n_estimators     = 1600,
    learning_rate    = 0.03,
    max_depth        = 8,
    subsample        = 0.9,
    colsample_bytree = 0.9,
    reg_lambda       = 1.0,
    tree_method      = "hist",        # or "gpu_hist" if CUDA available
    random_state     = 42
).fit(X_tr, y_tr)

joblib.dump(Q_model, MODEL_FILE)
print(f"✔  Saved Q-model →  {MODEL_FILE}")

# -----------------------------------------------------------------------------
# Diagnostics (optional)
# -----------------------------------------------------------------------------
y_pred_tr  = Q_model.predict(X_tr)
y_pred_val = Q_model.predict(X_val)
print("Q  train R² :", r2_score(y_tr,  y_pred_tr))
print("Q  val   R² :", r2_score(y_val, y_pred_val))
print("Q  val  RMSE:", mean_squared_error(
      y_val, y_pred_val))

# =============================================================================
# 5) Wrapper identical in signature to old V_fit
# =============================================================================
def Q_hat(batch_sa: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
    batch_sa : ndarray (N, 12)  — columns in this order:
        A, B, I, R_rad, Poly, T_s,
        err_P, err_Ts, int_err_P, int_err_T,
        I_in, Tc_scaled

    Returns
    -------
    ndarray (N,)  — discounted-return prediction.
    ***Maximise*** it.
    """
    return Q_model.predict(batch_sa)

# Demo (optional)
if __name__ == "__main__":
    demo = X_val[:1]
    print("Demo Q̂ :", Q_hat(demo)[0])
