"""
Train and Compare Models Script
Trains and evaluates ML models using Stratified 5-Fold Cross-Validation.
Prints Train vs Test comparison metrics (R², MAE, ±1 Level Accuracy) and Per-Class Accuracy.
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


def evaluate_models():
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    features_csv = os.path.join(data_dir, 'features.csv')

    if not os.path.exists(features_csv):
        print(f"Error: {features_csv} not found. Please run build_features.py first.")
        return None

    df = pd.read_csv(features_csv)
    print(f"Loaded features dataset: {len(df)} samples across 12 classes.")

    feature_cols = ['r', 'g', 'b', 'l', 'a', 'b_lab']
    X = df[feature_cols].values
    y = df['fan_score'].values

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    models = {
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
        'Linear Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', LinearRegression())
        ]),
        'Ridge Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', Ridge(alpha=1.0))
        ])
    }

    comparison_results = []
    per_class_results = []
    classes = sorted(np.unique(y))

    for name, pipeline in models.items():
        tr_r2, te_r2 = [], []
        tr_mae, te_mae = [], []
        tr_rmse, te_rmse = [], []

        all_true, all_pred = [], []

        for train_idx, val_idx in skf.split(X, y):
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_te, y_te = X[val_idx], y[val_idx]

            pipeline.fit(X_tr, y_tr)

            # Predict Train
            p_tr = pipeline.predict(X_tr)
            tr_r2.append(r2_score(y_tr, p_tr))
            tr_mae.append(mean_absolute_error(y_tr, p_tr))
            tr_rmse.append(np.sqrt(mean_squared_error(y_tr, p_tr)))

            # Predict Test (Validation)
            p_te = pipeline.predict(X_te)
            te_r2.append(r2_score(y_te, p_te))
            te_mae.append(mean_absolute_error(y_te, p_te))
            te_rmse.append(np.sqrt(mean_squared_error(y_te, p_te)))

            all_true.extend(y_te)
            all_pred.extend(p_te)

        comparison_results.append({
            'Model': name,
            'Train R2': round(np.mean(tr_r2), 4),
            'Test R2': round(np.mean(te_r2), 4),
            'Train MAE': round(np.mean(tr_mae), 4),
            'Test MAE': round(np.mean(te_mae), 4),
            'Train RMSE': round(np.mean(tr_rmse), 4),
            'Test RMSE': round(np.mean(te_rmse), 4)
        })

        # Calculate per-class regression error metrics on Test set
        all_true = np.array(all_true)
        all_pred = np.array(all_pred)

        for c in classes:
            mask = (all_true == c)
            preds = all_pred[mask]
            mae_c = mean_absolute_error(all_true[mask], preds)
            rmse_c = np.sqrt(mean_squared_error(all_true[mask], preds))
            
            per_class_results.append({
                'Model': name,
                'Class (Fan Score)': c,
                'Samples': int(np.sum(mask)),
                'Mean Pred': round(np.mean(preds), 2),
                'MAE': round(mae_c, 4),
                'RMSE': round(rmse_c, 4)
            })

    res_df = pd.DataFrame(comparison_results).sort_values(by='Test R2', ascending=False)
    per_class_df = pd.DataFrame(per_class_results)

    print("\n" + "=" * 90)
    print("MODEL TRAIN vs TEST COMPARISON RESULTS (Stratified 5-Fold Cross-Validation)")
    print("=" * 90)
    print(res_df.to_string(index=False))
    print("=" * 90)

    best_model_name = res_df.iloc[0]['Model']
    print(f"\nBest Performing Model: {best_model_name} (Test R^2 = {res_df.iloc[0]['Test R2']:.4f})")

    print(f"\n\n=== PER-CLASS REGRESSION ERROR BREAKDOWN FOR TOP MODEL ({best_model_name}) ===")
    print(per_class_df[per_class_df['Model'] == best_model_name].to_string(index=False))
    print("=" * 90)

    return res_df, per_class_df


if __name__ == '__main__':
    evaluate_models()
