import os
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, confusion_matrix
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Create plots directory
plots_dir = os.path.join(os.path.dirname(__file__), 'plots')
os.makedirs(plots_dir, exist_ok=True)

# Load dataset
base_dir = os.path.dirname(__file__)
autocrop_csv = os.path.join(base_dir, 'data', 'features_autocrop.csv')
features_csv = autocrop_csv if os.path.exists(autocrop_csv) else os.path.join(base_dir, 'data', 'features.csv')
df = pd.read_csv(features_csv)

feature_cols = ['r', 'g', 'b', 'l', 'a', 'b_lab']
X = df[feature_cols].values
y = df['fan_score'].values
classes = sorted(np.unique(y))

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Models dictionary
models = {
    'SVR (RBF Kernel)': Pipeline([('scaler', StandardScaler()), ('regressor', SVR(kernel='rbf', C=10.0, epsilon=0.1))]),
    'Gradient Boosting': Pipeline([('scaler', StandardScaler()), ('regressor', GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42))]),
    'Random Forest': Pipeline([('scaler', StandardScaler()), ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))]),
    'Linear Regression': Pipeline([('scaler', StandardScaler()), ('regressor', LinearRegression())]),
    'Ridge Regression': Pipeline([('scaler', StandardScaler()), ('regressor', Ridge(alpha=1.0))])
}

# Collect 5-fold cross-validation predictions for SVR and all models
all_true_svr = []
all_pred_svr = []
all_rounded_svr = []

model_performance = []

for name, pipeline in models.items():
    tr_r2, te_r2 = [], []
    tr_mae, te_mae = [], []

    all_t, all_p = [], []

    for train_idx, val_idx in skf.split(X, y):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_te, y_te = X[val_idx], y[val_idx]

        pipeline.fit(X_tr, y_tr)

        p_tr = pipeline.predict(X_tr)
        p_te = pipeline.predict(X_te)

        tr_r2.append(r2_score(y_tr, p_tr))
        te_r2.append(r2_score(y_te, p_te))
        tr_mae.append(mean_absolute_error(y_tr, p_tr))
        te_mae.append(mean_absolute_error(y_te, p_te))

        all_t.extend(y_te)
        all_p.extend(p_te)

    model_performance.append({
        'Model': name,
        'Test R2': np.mean(te_r2),
        'Test MAE': np.mean(te_mae)
    })

    if name == 'SVR (RBF Kernel)':
        all_true_svr = np.array(all_t)
        all_pred_svr = np.array(all_p)
        all_rounded_svr = np.round(all_pred_svr).astype(int)

# Set style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Tahoma']

# 1. Plot Confusion Matrix
plt.figure(figsize=(10, 8))
cm = confusion_matrix(all_true_svr, all_rounded_svr, labels=classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd', xticklabels=classes, yticklabels=classes, cbar=True)
plt.title('Confusion Matrix - SVR (RBF Kernel)\nActual vs Predicted Yolk Color Fan Class', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Predicted Yolk Color Fan Score', fontsize=12, fontweight='bold')
plt.ylabel('Actual Yolk Color Fan Score', fontsize=12, fontweight='bold')
plt.tight_layout()
cm_path = os.path.join(plots_dir, 'confusion_matrix.png')
plt.savefig(cm_path, dpi=300)
plt.close()
print(f"Saved: {cm_path}")

# 2. Plot Actual vs Predicted Scatter Plot
plt.figure(figsize=(9, 7))
plt.scatter(all_true_svr, all_pred_svr, alpha=0.6, color='#E1740A', edgecolors='k', linewidth=0.5, label='Egg Samples (N=647)')
plt.plot([4, 15], [4, 15], 'r--', linewidth=2, label='Ideal Prediction (y = x)')
r2_val = r2_score(all_true_svr, all_pred_svr)
mae_val = mean_absolute_error(all_true_svr, all_pred_svr)
plt.title(f'Actual vs Predicted Yolk Color Fan Score (SVR RBF Kernel)\nTest R² = {r2_val:.4f}, MAE = {mae_val:.4f}', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Actual Yolk Color Fan Score', fontsize=12, fontweight='bold')
plt.ylabel('Predicted Yolk Color Fan Score', fontsize=12, fontweight='bold')
plt.xticks(classes)
plt.yticks(classes)
plt.legend(fontsize=11)
plt.tight_layout()
scatter_path = os.path.join(plots_dir, 'actual_vs_predicted.png')
plt.savefig(scatter_path, dpi=300)
plt.close()
print(f"Saved: {scatter_path}")

# 3. Plot Residuals Distribution
residuals = all_pred_svr - all_true_svr
plt.figure(figsize=(9, 6))
sns.histplot(residuals, kde=True, color='#FBA919', bins=25, edgecolor='black')
plt.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error Line')
plt.title('Prediction Error Residuals Distribution (Predicted - Actual)', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Prediction Error (Fan Score Levels)', fontsize=12, fontweight='bold')
plt.ylabel('Frequency (Egg Samples)', fontsize=12, fontweight='bold')
plt.legend(fontsize=11)
plt.tight_layout()
residuals_path = os.path.join(plots_dir, 'residuals_distribution.png')
plt.savefig(residuals_path, dpi=300)
plt.close()
print(f"Saved: {residuals_path}")

# 4. Plot Model Comparison Bar Chart
perf_df = pd.DataFrame(model_performance).sort_values('Test R2', ascending=True)
fig, ax1 = plt.subplots(figsize=(10, 6))

bars = ax1.barh(perf_df['Model'], perf_df['Test R2'], color='#FBA919', edgecolor='black', alpha=0.85)
ax1.set_xlabel('Test R² Score (Higher is Better)', fontsize=12, fontweight='bold', color='#B45309')
ax1.set_xlim(0.75, 0.95)

for bar in bars:
    width = bar.get_width()
    ax1.text(width + 0.003, bar.get_y() + bar.get_height()/2, f'{width:.4f}', ha='left', va='center', fontsize=10, fontweight='bold')

plt.title('ML Model Comparison (Stratified 5-Fold Cross-Validation)', fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
bar_path = os.path.join(plots_dir, 'model_comparison_bar.png')
plt.savefig(bar_path, dpi=300)
plt.close()
print(f"Saved: {bar_path}")

print("All 4 evaluation plots generated successfully!")
