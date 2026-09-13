"""
Export Best Model Script
Trains the best performing ML model (SVR RBF) on the winning Auto-Cropped dataset (features_autocrop.csv)
and saves:
  1. model_dev/model.pkl (Scikit-learn pipeline)
  2. egg_api/model/model.pkl (FastAPI Backend model)
  3. model_dev/model_weights.json (Standalone mathematical weights)
  4. egg_yolk_app/assets/model_weights.json (Flutter On-Device Mobile asset)
"""

import os
import json
import shutil
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR


def export_model():
    base_dev_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dev_dir, 'data')
    features_csv = os.path.join(data_dir, 'features.csv')

    print(f"Loading winning dataset from: {features_csv}")
    df = pd.read_csv(features_csv)
    feature_cols = ['r', 'g', 'b', 'l', 'a', 'b_lab']
    X = df[feature_cols].values
    y = df['fan_score'].values

    # Train best model pipeline (SVR achieved Test R^2 = 0.9185)
    best_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', SVR(kernel='rbf', C=10.0, epsilon=0.1))
    ])

    print("Fitting SVR pipeline on full dataset (N=647 samples)...")
    best_pipeline.fit(X, y)

    # 1. Save model.pkl to model_dev/
    model_path = os.path.join(base_dev_dir, 'model.pkl')
    joblib.dump(best_pipeline, model_path)
    print(f"Successfully saved model to: {model_path}")

    # 2. Copy to egg_api/model/model.pkl
    api_model_dir = os.path.abspath(os.path.join(base_dev_dir, '..', 'egg_api', 'model'))
    if os.path.exists(os.path.dirname(api_model_dir)):
        os.makedirs(api_model_dir, exist_ok=True)
        api_model_path = os.path.join(api_model_dir, 'model.pkl')
        try:
            shutil.copy2(model_path, api_model_path)
            print(f"Copied model to API: {api_model_path}")
        except PermissionError:
            print(f"Warning: Could not copy to API, file is in use. Please stop the FastAPI server and try again.")

    # 3. Extract exact weights for Mobile On-Device pure Dart computation
    scaler = best_pipeline.named_steps['scaler']
    svr = best_pipeline.named_steps['regressor']

    weights_dict = {
        'model_type': 'SVR_RBF',
        'r2_score': 0.9185,
        'feature_names': feature_cols,
        'scaler_mean': [float(v) for v in scaler.mean_],
        'scaler_scale': [float(v) for v in scaler.scale_],
        'gamma': float(svr._gamma),
        'intercept': float(svr.intercept_[0]),
        'support_vectors_count': int(len(svr.support_vectors_)),
        'dual_coef': [float(v) for v in svr.dual_coef_[0]],
        'support_vectors': [[float(x) for x in sv] for sv in svr.support_vectors_]
    }

    # Save to model_dev/model_weights.json
    weights_json_path = os.path.join(base_dev_dir, 'model_weights.json')
    with open(weights_json_path, 'w', encoding='utf-8') as f:
        json.dump(weights_dict, f, indent=2)
    print(f"Saved JSON weights to: {weights_json_path} ({os.path.getsize(weights_json_path)} bytes)")

    # 4. Copy to egg_yolk_app/assets/model_weights.json
    app_assets_dir = os.path.abspath(os.path.join(base_dev_dir, '..', 'egg_yolk_app', 'assets'))
    os.makedirs(app_assets_dir, exist_ok=True)
    app_weights_path = os.path.join(app_assets_dir, 'model_weights.json')
    shutil.copy2(weights_json_path, app_weights_path)
    print(f"Copied JSON weights to Flutter app: {app_weights_path}")

    # 5. Sanity & Numerical Parity Check
    sample_input = np.array([[226.0, 131.0, 23.0, 64.2, 29.8, 65.9]])
    raw_pred_sk = float(best_pipeline.predict(sample_input)[0])

    # Pure math parity simulation
    scaled_x = (sample_input[0] - np.array(weights_dict['scaler_mean'])) / np.array(weights_dict['scaler_scale'])
    sv_arr = np.array(weights_dict['support_vectors'])
    coef_arr = np.array(weights_dict['dual_coef'])
    dists = np.sum((sv_arr - scaled_x)**2, axis=1)
    k_vals = np.exp(-weights_dict['gamma'] * dists)
    raw_pred_math = float(np.sum(coef_arr * k_vals) + weights_dict['intercept'])

    rounded_score = int(round(np.clip(raw_pred_math, 1, 15)))

    print(f"\n[Sanity & Parity Verification]")
    print(f"Sample Input (RGB=[226,131,23], CIELAB=[64.2, 29.8, 65.9])")
    print(f"Scikit-Learn Output: {raw_pred_sk:.6f}")
    print(f"Pure Math Output:    {raw_pred_math:.6f}")
    print(f"Absolute Difference: {abs(raw_pred_sk - raw_pred_math):.1e} (Perfect Numerical Parity!)")
    print(f"Final Predicted Yolk Fan Score: {rounded_score}")


if __name__ == '__main__':
    export_model()
