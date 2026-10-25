import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
import joblib
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent
def train():
    # 1. Load & parse
    input_path = PROJECT_ROOT / 'data' / 'all_data_READY.csv'
    figure_path = PROJECT_ROOT / 'figures' / 'statistics'

    df = pd.read_csv(input_path)

    df['Datetime'] = pd.to_datetime(df['Datetime'], format='%Y-%m-%d %H:%M:%S')
    df = df.drop_duplicates(['Datetime', 'Company']).sort_values(['Company', 'Datetime'])

    # 2. Calendar features
    df['Hour'] = df['Datetime'].dt.hour
    df['Dayofweek'] = df['Datetime'].dt.dayofweek
    df['Is_weekend'] = (df['Dayofweek'] >= 5).astype(int)
    df['Month'] = df['Datetime'].dt.month
    df['Dayofyear'] = df['Datetime'].dt.dayofyear

    # 3. Encode company
    df['Company_code'] = df['Company'].astype('category').cat.codes

    # 4. Lag & rolling features (per company, leak-free)
    g = df.groupby('Company')['MW']
    for lag in [1, 24, 168]:
        df[f'Lag_{lag}'] = g.shift(lag)
    df['Roll_mean_24'] = g.transform(lambda s: s.shift(1).rolling(24).mean())
    df['Roll_std_24'] = g.transform(lambda s: s.shift(1).rolling(24).std())

    df = df.dropna()

    # 5. Define X / y and time-based split
    features = ['Hour', 'Dayofweek', 'Is_weekend', 'Month', 'Dayofyear', 'Company_code',
                'Lag_1', 'Lag_24', 'Lag_168', 'Roll_mean_24', 'Roll_std_24']
    df = df.sort_values('Datetime')
    split = df['Datetime'].quantile(0.8)
    train, test = df[df['Datetime'] <= split], df[df['Datetime'] > split]
    X_train, y_train = train[features], train['MW']
    X_test, y_test = test[features], test['MW']

    # 6. Hyperparameter tuning with time-aware CV
    """
    param_dist = {
        'n_estimators': [100, 200],  # drop 400
        'max_depth': [10, 20, 30],  # drop None
        'min_samples_leaf': [2, 5],  # drop 1
        'max_features': ['sqrt', 0.5],  # drop 1.0
    }
    """
    param_dist = {
        'n_estimators': [100, 150],  # Lower the upper limit
        'max_depth': [10, 15],  # No longer use 20/30
        'min_samples_leaf': [3, 5],  # Slightly increase leaf minimum samples
        'max_features': ['sqrt', 0.5],
    }
    search = RandomizedSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=1),  # inner: serial
        param_dist, n_iter=15, cv=TimeSeriesSplit(n_splits=5),
        scoring='neg_mean_absolute_error', random_state=42,
        n_jobs=-3, # outer: parallel
        verbose=2
    )
    #print(X_train.shape, X_train.dtypes, X_train.nbytes / 1e6, "MB")
    search.fit(X_train, y_train)
    rf = search.best_estimator_

    # 7. Evaluate
    pred = rf.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4), gridspec_kw={'width_ratios': [1, 3]})

    # Metrics table
    axes[0].axis('off')
    table = axes[0].table(
        cellText=[[f'{mae:.3f}'], [f'{rmse:.3f}'], [f'{r2:.3f}']],
        rowLabels=['MAE (MW)', 'RMSE (MW)', 'R²'],
        colLabels=['Test score'],
        loc='center', cellLoc='center')
    table.scale(1, 2)
    axes[0].set_title('Evaluation Metrics')

    # Actual vs. predicted over time (first 7 days of test set for readability)
    plot_df = test.assign(pred=pred).sort_values('Datetime')
    subset = plot_df.head(168)
    axes[1].plot(subset['Datetime'], subset['MW'], label='Actual')
    axes[1].plot(subset['Datetime'], subset['pred'], label='Predicted', alpha=0.8)
    axes[1].set_title('Actual vs. Predicted Consumption (first 7 days of test set)')
    axes[1].set_xlabel('Datetime')
    axes[1].set_ylabel('MW')
    axes[1].legend()

    plt.tight_layout()
    merged_path = figure_path / 'random_forests_table.png'
    #plt.show()
    fig.savefig(merged_path)

    # 8. Feature importance / selection
    imp = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42)

    imp_df = (pd.DataFrame({'feature': features,
                            'importance': imp.importances_mean,
                            'std': imp.importances_std})
              .sort_values('importance'))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(imp_df['feature'], imp_df['importance'], xerr=imp_df['std'], capsize=3)
    ax.set_title('Permutation Feature Importance (test set)')
    ax.set_xlabel('Mean decrease in score when feature is shuffled')
    plt.tight_layout()
    merged_path = figure_path / 'random_forests_permutation_importance.png'
    fig.savefig(merged_path)

    residuals = y_test - pred

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].scatter(pred, residuals, s=5, alpha=0.4)
    axes[0].axhline(0, color='red', linestyle='--')
    axes[0].set_title('Residuals vs. Predicted')
    axes[0].set_xlabel('Predicted MW')
    axes[0].set_ylabel('Residual (Actual - Predicted)')

    axes[1].hist(residuals, bins=50)
    axes[1].set_title('Residual Distribution')
    axes[1].set_xlabel('Residual (MW)')

    plt.tight_layout()
    merged_path = figure_path / 'random_forests_residuals.png'
    fig.savefig(merged_path)

    # Save
    joblib.dump(rf, 'rf_energy_model.joblib')

def check_size():
    df = pd.read_csv("data/all_data_READY.csv")
    read_length=len(df)
    df = pd.read_csv("data/cleaned/all_data_relabelled_CLEANED.csv")
    cleaned_relabeled = len(df)
    df = pd.read_csv("data/all_data_relabelled.csv")
    relabeled = len(df)

    df = pd.read_csv("data/all_data.csv")
    all = len(df)

    print(f"Ready: {read_length}\nCleaned Relabeled: {cleaned_relabeled}\nRelabeled: {relabeled}\nAll: {all}")
    print(df.info())
    print(df.head())

if __name__ == "__main__":
    #train()
    check_size()