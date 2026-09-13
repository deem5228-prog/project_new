"""
Compare Crop Experiments Script
Trains and evaluates ML models on both Baseline (Original) and Auto-Cropped Datasets
using identical Stratified 5-Fold Cross-Validation (random_state=42).
Prints a side-by-side comparison table to determine which dataset yields higher accuracy.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def get_models():
    return {
        'SVR (RBF Kernel)': Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', SVR(kernel='rbf', C=10.0, epsilon=0.1))
        ]),
        'Gradient Boosting': Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42))
        ]),
        'Random Forest': Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
        ]),
        'Ridge Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', Ridge(alpha=1.0))
        ]),
        'Linear Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', LinearRegression())
        ])
    }


def evaluate_dataset(csv_path, dataset_name):
    df = pd.read_csv(csv_path)
    feature_cols = ['r', 'g', 'b', 'l', 'a', 'b_lab']
    X = df[feature_cols].values
    y = df['fan_score'].values

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models = get_models()
    results = {}

    for name, pipeline in models.items():
        te_r2, te_mae, te_rmse = [], [], []
        all_true, all_pred = [], []

        for train_idx, val_idx in skf.split(X, y):
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_te, y_te = X[val_idx], y[val_idx]

            pipeline.fit(X_tr, y_tr)
            p_te = pipeline.predict(X_te)

            te_r2.append(r2_score(y_te, p_te))
            te_mae.append(mean_absolute_error(y_te, p_te))
            te_rmse.append(np.sqrt(mean_squared_error(y_te, p_te)))

            all_true.extend(y_te)
            all_pred.extend(p_te)

        all_true = np.array(all_true)
        all_pred = np.array(all_pred)
        pm1_acc = np.mean(np.abs(np.round(all_pred) - all_true) <= 1) * 100.0

        results[name] = {
            'R2': np.mean(te_r2),
            'MAE': np.mean(te_mae),
            'RMSE': np.mean(te_rmse),
            'PM1_Acc': pm1_acc
        }

    return results


def run_comparison():
    base_dev_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dev_dir, 'data')
    orig_csv = os.path.join(data_dir, 'features.csv')
    autocrop_csv = os.path.join(data_dir, 'features_autocrop.csv')

    if not os.path.exists(orig_csv):
        print(f"Error: {orig_csv} not found.")
        return

    if not os.path.exists(autocrop_csv):
        print(f"Error: {autocrop_csv} not found. Please run build_features_autocrop.py first.")
        return

    print("Evaluating Baseline (Original Manual Crop)...")
    orig_results = evaluate_dataset(orig_csv, "Original Manual Crop")

    print("Evaluating Auto-Cropped Dataset (HSV Color Segmentation)...")
    autocrop_results = evaluate_dataset(autocrop_csv, "Auto-Cropped Dataset")

    # Reconfigure stdout for utf-8 on Windows
    import sys
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

    # Build comparison table
    rows = []
    models = get_models()
    for name in models.keys():
        o = orig_results[name]
        a = autocrop_results[name]

        diff_r2 = a['R2'] - o['R2']
        winner = "Auto-Crop (Win)" if a['R2'] > o['R2'] else "Original (Win)"

        rows.append({
            'Model': name,
            'Original R2': f"{o['R2']:.4f}",
            'Auto-Crop R2': f"{a['R2']:.4f}",
            'Diff R2': f"{diff_r2:+.4f}",
            'Original MAE': f"{o['MAE']:.4f}",
            'Auto-Crop MAE': f"{a['MAE']:.4f}",
            'Original +-1 Acc': f"{o['PM1_Acc']:.1f}%",
            'Auto-Crop +-1 Acc': f"{a['PM1_Acc']:.1f}%",
            'Winner': winner
        })

    comp_df = pd.DataFrame(rows)

    print("\n" + "=" * 110)
    print("A/B EXPERIMENT COMPARISON RESULTS: ORIGINAL (MANUAL) vs AUTO-DETECT CROP")
    print("Stratified 5-Fold Cross-Validation (Seed = 42)")
    print("=" * 110)
    print(comp_df.to_string(index=False))
    print("=" * 110)

    # Save comparison to CSV
    out_csv = os.path.join(data_dir, 'crop_comparison_results.csv')
    comp_df.to_csv(out_csv, index=False)
    print(f"\nSaved comparison summary table to: {out_csv}")


if __name__ == '__main__':
    run_comparison()
