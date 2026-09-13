"""
Generate All Publication-Grade Evaluation Plots
Based on the new Center Circular Masking features (R2 = 0.9175)
"""

import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Set styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11

# Paths setup
base_dir = os.path.dirname(__file__)
autocrop_csv = os.path.join(base_dir, 'data', 'features_autocrop.csv')
data_csv = autocrop_csv if os.path.exists(autocrop_csv) else os.path.join(base_dir, 'data', 'features.csv')
plots_dir = os.path.join(base_dir, 'plots')
os.makedirs(plots_dir, exist_ok=True)

brain_dir = r'C:\Users\Admin\.gemini\antigravity\brain\baddf511-103e-4c9f-926d-8161adce50ce'

# 1. Load Data
df = pd.read_csv(data_csv)
feature_cols = ['r', 'g', 'b', 'l', 'a', 'b_lab']
X = df[feature_cols].values
y = df['fan_score'].values
classes = sorted(np.unique(y))

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    'SVR (RBF Kernel)': Pipeline([('scaler', StandardScaler()), ('regressor', SVR(kernel='rbf', C=10.0, epsilon=0.1))]),
    'Random Forest': Pipeline([('scaler', StandardScaler()), ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))]),
    'Gradient Boosting': Pipeline([('scaler', StandardScaler()), ('regressor', GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42))]),
    'Linear Regression': Pipeline([('scaler', StandardScaler()), ('regressor', LinearRegression())]),
    'Ridge Regression': Pipeline([('scaler', StandardScaler()), ('regressor', Ridge(alpha=1.0))])
}

# Run Cross-Validation
comparison_data = []
all_y_true = []
all_y_pred_svr = []

for name, pipe in models.items():
    tr_r2, te_r2 = [], []
    tr_mae, te_mae = [], []
    tr_rmse, te_rmse = [], []
    
    for tr_idx, te_idx in skf.split(X, y):
        pipe.fit(X[tr_idx], y[tr_idx])
        p_tr = pipe.predict(X[tr_idx])
        p_te = pipe.predict(X[te_idx])
        
        tr_r2.append(r2_score(y[tr_idx], p_tr))
        te_r2.append(r2_score(y[te_idx], p_te))
        tr_mae.append(mean_absolute_error(y[tr_idx], p_tr))
        te_mae.append(mean_absolute_error(y[te_idx], p_te))
        tr_rmse.append(np.sqrt(mean_squared_error(y[tr_idx], p_tr)))
        te_rmse.append(np.sqrt(mean_squared_error(y[te_idx], p_te)))
        
        if name == 'SVR (RBF Kernel)':
            all_y_true.extend(y[te_idx])
            all_y_pred_svr.extend(p_te)
            
    comparison_data.append({
        'Model': name,
        'Train R2': np.mean(tr_r2),
        'Test R2': np.mean(te_r2),
        'Train MAE': np.mean(tr_mae),
        'Test MAE': np.mean(te_mae),
        'Train RMSE': np.mean(tr_rmse),
        'Test RMSE': np.mean(te_rmse)
    })

comp_df = pd.DataFrame(comparison_data)
all_y_true = np.array(all_y_true)
all_y_pred_svr = np.array(all_y_pred_svr)
residuals = all_y_pred_svr - all_y_true

print("Completed 5-Fold Cross-Validation. Generating plots...")

# =========================================================================
# PLOT 1: Model Comparison Bar Chart (Test R2, MAE, RMSE)
# =========================================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
palette = ['#10B981', '#3B82F6', '#8B5CF6', '#F59E0B', '#EF4444']

# 1.1 R2 Score
axes[0].bar(comp_df['Model'], comp_df['Test R2'], color=palette, edgecolor='#333', width=0.55)
axes[0].set_title('Test R² Score (Higher is Better)', fontweight='bold', fontsize=13)
axes[0].set_ylim(0.75, 0.96)
axes[0].set_ylabel('R² Score')
axes[0].tick_params(axis='x', rotation=25)
for i, v in enumerate(comp_df['Test R2']):
    axes[0].text(i, v + 0.005, f"{v:.4f}", ha='center', fontweight='bold', fontsize=10)

# 1.2 MAE
axes[1].bar(comp_df['Model'], comp_df['Test MAE'], color=palette, edgecolor='#333', width=0.55)
axes[1].set_title('Test MAE Error (Lower is Better)', fontweight='bold', fontsize=13)
axes[1].set_ylim(0, 1.05)
axes[1].set_ylabel('Mean Absolute Error (Fan Units)')
axes[1].tick_params(axis='x', rotation=25)
for i, v in enumerate(comp_df['Test MAE']):
    axes[1].text(i, v + 0.02, f"{v:.4f}", ha='center', fontweight='bold', fontsize=10)

# 1.3 RMSE
axes[2].bar(comp_df['Model'], comp_df['Test RMSE'], color=palette, edgecolor='#333', width=0.55)
axes[2].set_title('Test RMSE Error (Lower is Better)', fontweight='bold', fontsize=13)
axes[2].set_ylim(0, 1.35)
axes[2].set_ylabel('Root Mean Squared Error (Fan Units)')
axes[2].tick_params(axis='x', rotation=25)
for i, v in enumerate(comp_df['Test RMSE']):
    axes[2].text(i, v + 0.03, f"{v:.4f}", ha='center', fontweight='bold', fontsize=10)

plt.tight_layout()
p1_path = os.path.join(plots_dir, 'model_comparison_bar.png')
plt.savefig(p1_path, dpi=300, bbox_inches='tight')
plt.close()

# =========================================================================
# PLOT 2: Actual vs Predicted Scatter Plot (SVR)
# =========================================================================
plt.figure(figsize=(8, 7.5))
# Jitter slightly for visual density
jitter_x = all_y_true + np.random.normal(0, 0.08, len(all_y_true))
plt.scatter(jitter_x, all_y_pred_svr, color='#2563EB', alpha=0.6, edgecolors='none', s=45, label='Egg Yolk Samples (N=647)')

# Identity Line
min_val, max_val = 3.5, 15.5
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Ideal 1:1 Identity Line (y = x)')

# Stats Box
svr_r2 = comp_df.loc[comp_df['Model'] == 'SVR (RBF Kernel)', 'Test R2'].values[0]
svr_mae = comp_df.loc[comp_df['Model'] == 'SVR (RBF Kernel)', 'Test MAE'].values[0]
svr_rmse = comp_df.loc[comp_df['Model'] == 'SVR (RBF Kernel)', 'Test RMSE'].values[0]

stats_text = (
    f"SVR (RBF Kernel)\n"
    f"--------------------\n"
    f"Test R² = {svr_r2:.4f} ({svr_r2*100:.2f}%)\n"
    f"Test MAE = {svr_mae:.4f} Fan Score\n"
    f"Test RMSE = {svr_rmse:.4f} Fan Score\n"
    f"Samples = {len(all_y_true)} (5-Fold CV)"
)
plt.text(4.0, 13.0, stats_text, fontsize=11, fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.6', facecolor='#F3F4F6', edgecolor='#9CA3AF', alpha=0.95))

plt.title('Actual vs Predicted Egg Yolk Color Fan Score (SVR RBF)', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Actual Fan Score (Ground Truth Class 4–15)', fontsize=11, fontweight='bold')
plt.ylabel('Predicted Score (Continuous Model Output)', fontsize=11, fontweight='bold')
plt.xlim(min_val, max_val)
plt.ylim(min_val, max_val)
plt.xticks(range(4, 16))
plt.yticks(range(4, 16))
plt.legend(loc='lower right', frameon=True)
plt.tight_layout()

p2_path = os.path.join(plots_dir, 'actual_vs_predicted.png')
plt.savefig(p2_path, dpi=300, bbox_inches='tight')
plt.close()

# =========================================================================
# PLOT 3: Residuals Distribution
# =========================================================================
plt.figure(figsize=(9, 5.5))
sns.histplot(residuals, kde=True, color='#059669', bins=30, edgecolor='black', alpha=0.65)
plt.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error (Residual = 0)')
plt.axvline(np.mean(residuals), color='blue', linestyle=':', linewidth=2, label=f'Mean Error = {np.mean(residuals):.3f}')

plt.title('Residuals (Prediction Error) Distribution - SVR Model', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Residual Error (Predicted - Actual)', fontsize=11, fontweight='bold')
plt.ylabel('Sample Count (Frequency)', fontsize=11, fontweight='bold')

res_text = (
    f"Mean Error: {np.mean(residuals):.3f}\n"
    f"Std Dev (σ): {np.std(residuals):.3f}\n"
    f"Continuous Error (≤ ±1.0): {np.mean(np.abs(residuals) <= 1.0)*100:.1f}%\n"
    f"Rounded Level (≤ ±1 Level): {np.mean(np.abs(np.round(all_y_pred_svr) - all_y_true) <= 1)*100:.1f}%"
)
plt.text(0.60, 0.72, res_text, transform=plt.gca().transAxes, fontsize=10.5,
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#ECFDF5', edgecolor='#10B981', alpha=0.95))

plt.legend(loc='upper left', frameon=True)
plt.tight_layout()

p3_path = os.path.join(plots_dir, 'residuals_distribution.png')
plt.savefig(p3_path, dpi=300, bbox_inches='tight')
plt.close()

# =========================================================================
# PLOT 4: Feature Correlation Matrix Heatmap
# =========================================================================
plt.figure(figsize=(8, 6.5))
corr = df[['r', 'g', 'b', 'l', 'a', 'b_lab', 'fan_score']].corr()
rename_cols = {'r':'R', 'g':'G', 'b':'B', 'l':'L*', 'a':'a*', 'b_lab':'b*', 'fan_score':'Fan Score'}
corr = corr.rename(index=rename_cols, columns=rename_cols)

sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', cbar=True, vmin=-1, vmax=1,
            linewidths=1, linecolor='white', annot_kws={'fontsize': 11, 'fontweight': 'bold'})
plt.title('Color Feature Correlation Matrix (CIELAB & RGB vs Fan Score)', fontsize=13, fontweight='bold', pad=12)
plt.tight_layout()

p4_path = os.path.join(plots_dir, 'feature_correlation_matrix.png')
plt.savefig(p4_path, dpi=300, bbox_inches='tight')
plt.close()

# =========================================================================
# PLOT 5: Per-Class MAE and RMSE Breakdown
# =========================================================================
per_class_mae = []
per_class_rmse = []
for c in classes:
    m = (all_y_true == c)
    per_class_mae.append(mean_absolute_error(all_y_true[m], all_y_pred_svr[m]))
    per_class_rmse.append(np.sqrt(mean_squared_error(all_y_true[m], all_y_pred_svr[m])))

plt.figure(figsize=(11, 5.5))
x_idx = np.arange(len(classes))
width = 0.38

plt.bar(x_idx - width/2, per_class_mae, width=width, label='MAE (Mean Absolute Error)', color='#3B82F6', edgecolor='#1E3A8A')
plt.bar(x_idx + width/2, per_class_rmse, width=width, label='RMSE (Root Mean Squared Error)', color='#EF4444', edgecolor='#991B1B')

plt.title('Per-Class Error Breakdown (MAE & RMSE across Classes 4–15)', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Egg Yolk Color Fan Score Class', fontsize=11, fontweight='bold')
plt.ylabel('Error (Fan Units)', fontsize=11, fontweight='bold')
plt.xticks(x_idx, [f"Class {c}" for c in classes])
plt.ylim(0, 1.4)

for i in range(len(classes)):
    plt.text(i - width/2, per_class_mae[i] + 0.03, f"{per_class_mae[i]:.2f}", ha='center', fontsize=9, fontweight='bold')
    plt.text(i + width/2, per_class_rmse[i] + 0.03, f"{per_class_rmse[i]:.2f}", ha='center', fontsize=9, fontweight='bold')

plt.legend(frameon=True)
plt.tight_layout()

p5_path = os.path.join(plots_dir, 'per_class_mae_rmse.png')
plt.savefig(p5_path, dpi=300, bbox_inches='tight')
plt.close()

# Copy to brain artifact directory for display
for p in [p1_path, p2_path, p3_path, p4_path, p5_path]:
    dest = os.path.join(brain_dir, os.path.basename(p))
    shutil.copy2(p, dest)

print("Successfully generated all 5 updated plots!")
