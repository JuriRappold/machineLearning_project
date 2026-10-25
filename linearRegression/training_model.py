import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from linear_regression import (
    LinearRegressionScratch, mae, rmse, r2,
)
from feature_extraction import (
    chronological_split, build_features, build_with_interactions,
)

# --- Synthetic energy data (replace with your real loader) ---
"""
rng = np.random.default_rng(0)
n_hours = 24 * 60  # 60 days of hourly data
companies = np.array(["A", "B", "C"])
rows = []
for t in range(n_hours):
    ts = pd.Timestamp("2024-01-01") + pd.Timedelta(hours=t)
    co = companies[rng.integers(0, len(companies))]
    base = {"A": 50, "B": 30, "C": 80}[co]
    hour = ts.hour
    seasonal = 10 * np.sin(2 * np.pi * hour / 24)
    weekly = 5 * (1 if ts.dayofweek >= 5 else 0)
    mw = base + seasonal + weekly + rng.normal(0, 1.0)
    rows.append([ts.strftime("%Y-%m-%d"), ts.strftime("%H:00:00"), co, mw])
df = pd.DataFrame(rows, columns=["date", "time", "company", "MW"])
"""
def train_linea_regression():
    df = pd.read_csv("../data/all_data_READY.csv")

    DATETIME = "Datetime"
    COMPANY = "Company"
    MEGAWATTS = "MW"
    ORIGINAL_COLUMNS = [COMPANY, DATETIME, MEGAWATTS]
    print(len(df))
    df[DATETIME] = pd.to_datetime(df[DATETIME], errors="raise")


    # --- Chronological split ---
    train_df, test_df = chronological_split(df, test_frac=0.2)

    # --- Plain features (no interactions) ---
    X_tr, y_tr, enc, scl = build_features(train_df, fit=True)
    X_te, y_te, _, _ = build_features(test_df, encoder=enc, scaler=scl, fit=False)

    # Normal equation
    """
    m_norm = LinearRegressionScratch(method="normal", l2=1e-6).fit(X_tr, y_tr)
    pred_te = m_norm.predict(X_te)
    print(f"[normal] MAE={mae(y_te, pred_te):.3f} RMSE={rmse(y_te, pred_te):.3f} R²={r2(y_te, pred_te):.3f}")
    joblib.dump(m_norm, "lr_normal_energy_model.joblib")
    """

    # Gradient descent (scaled features -> lr can be modest)
    m_gd = LinearRegressionScratch(method="gd", l2=1e-6, lr=1e-2, n_iters=20000, tol=1e-10) \
        .fit(X_tr, y_tr)
    pred_te = m_gd.predict(X_te)
    print(f"[gd    ] MAE={mae(y_te, pred_te):.3f} RMSE={rmse(y_te, pred_te):.3f} R²={r2(y_te, pred_te):.3f}")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(m_gd.loss_history_)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Regularized MSE loss")
    ax.set_title("Gradient descent convergence")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('../figures/statistics/lr_gradient_stats.png')
    joblib.dump(m_gd, "lr_gradient_energy_model.joblib")


    # --- Interactions variant: bump l2 (much larger feature matrix) ---
    """
    Xi_tr, yi_tr, enc2, scl2 = build_with_interactions(train_df, fit=True)
    Xi_te, yi_te, _, _ = build_with_interactions(test_df, encoder=enc2, scaler=scl2, fit=False)
    m_int = LinearRegressionScratch(method="normal", l2=5.0).fit(Xi_tr, yi_tr)
    pred_te = m_int.predict(Xi_te)
    print(f"[inter ] MAE={mae(yi_te, pred_te):.3f} RMSE={rmse(yi_te, pred_te):.3f} R²={r2(yi_te, pred_te):.3f}")
    joblib.dump(m_int, 'lr_interactive_energy_model.joblib')
    """