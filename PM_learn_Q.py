# =============================================================================
# 0) Imports
# =============================================================================
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor
import joblib

CSV_PATH    = "paper_machine_batch_log_scaled.csv"   #  new Monte-Carlo dataset
MODEL_FILE  = "q_discounted.joblib"
SCALER_FILE = "q_scaler.joblib"
GAMMA       = 0.9         # discount factor

# =============================================================================
# 1) Load log
# =============================================================================
df = pd.read_csv(CSV_PATH)

# =============================================================================
# 2) Discounted return  G_t = Σ γᵏ r_{t+k}
# =============================================================================
df = df.sort_values(['run', 'step']).reset_index(drop=True)

disc_return = np.zeros(len(df))
for run_id, grp in df.groupby('run', sort=False):
    g = 0.0
    for idx in grp.index[::-1]:            # iterate backwards
        g = df.at[idx, 'reward'] + GAMMA * g
        disc_return[idx] = g
df['disc_return'] = disc_return

# =============================================================================
# 3) Features   s_t  +  a_t
# =============================================================================
state_cols  = [
    "Y1", "Y2", "Y3",                # current outputs
    "e1", "e2", "e3",                # errors
    "int1", "int2", "int3", "int4"   # integrator states
]
action_cols = ["U1", "U2", "U3", "U4"]
sa_cols     = state_cols + action_cols

X = df[sa_cols].values.astype(np.float32)
y = df['disc_return'].values.astype(np.float32)

# ----------------------------------------------------------------------------- 
# scale *X* (states + actions) — leave *y* untouched
# -----------------------------------------------------------------------------
X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.10, random_state=42, shuffle=True
)

scaler = StandardScaler().fit(X_tr)
X_tr   = scaler.transform(X_tr)
X_val  = scaler.transform(X_val)

# =============================================================================
# 4) XGBoost Q-function
# =============================================================================
Q_model = XGBRegressor(
    objective        = "reg:squarederror",
    n_estimators     = 1200,
    learning_rate    = 0.02,
    max_depth        = 12,
    subsample        = 0.9,
    colsample_bytree = 0.9,
    reg_lambda       = 1.0,
    tree_method      = "hist",
    random_state     = 43
).fit(X_tr, y_tr)

joblib.dump(Q_model,  MODEL_FILE)
joblib.dump(scaler,    SCALER_FILE)
print(f"✔  Saved Q-model  →  {MODEL_FILE}")
print(f"✔  Saved scaler   →  {SCALER_FILE}")

# ----------------------------------------------------------------------------- 
# Diagnostics
# -----------------------------------------------------------------------------
y_pred_tr  = Q_model.predict(X_tr)
y_pred_val = Q_model.predict(X_val)
print("Q  train R² :", r2_score(y_tr,  y_pred_tr))
print("Q  val   R² :", r2_score(y_val, y_pred_val))
print("Q  val  RMSE:", mean_squared_error(y_val, y_pred_val))

# =============================================================================
# 5) Inference helper (identical signature as before)
# =============================================================================
def Q_hat(batch_sa: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
    batch_sa : ndarray (N, 14)  — columns in this order:
        Y1,Y2,Y3, e1,e2,e3, int1..int4, U1..U4   (un-scaled)

    Returns
    -------
    ndarray (N,)  — discounted-return prediction (higher is better)
    """
    batch_sa_scaled = scaler.transform(batch_sa.astype(np.float32))
    return Q_model.predict(batch_sa_scaled)

# Demo
if __name__ == "__main__":
    demo = X_val[:1]            # already scaled, just for quick check
    print("Demo Q̂ (scaled input) :", Q_model.predict(demo)[0])
