import os
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Load real dataset
features_csv = r'c:\Users\Admin\Downloads\project_code\egg_yolk_project\model_dev\data\features.csv'
df = pd.read_csv(features_csv)

# Use redness component (a*) as primary X axis vs Yolk Fan Score (y) for 2D visualization
X_raw = df[['a']].values  # Redness a* feature
y = df['fan_score'].values

# Fit 1D SVR for clear visual mapping
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

svr = SVR(kernel='rbf', C=10.0, epsilon=0.1)
svr.fit(X_scaled, y)

# Generate smooth prediction line across a* range
x_grid_raw = np.linspace(X_raw.min(), X_raw.max(), 300).reshape(-1, 1)
x_grid_scaled = scaler.transform(x_grid_raw)
y_pred_line = svr.predict(x_grid_scaled)

# Get Support Vector indices and points
sv_indices = svr.support_
sv_x = X_raw[sv_indices].ravel()
sv_y = y[sv_indices]

# Epsilon tube bounds (epsilon = 0.1)
epsilon = 0.1
tube_upper = y_pred_line + epsilon
tube_lower = y_pred_line - epsilon

# Plotting
plt.figure(figsize=(11, 7))
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# 1. Plot Epsilon Tube Band (Translucent Amber Shading)
plt.fill_between(x_grid_raw.ravel(), tube_lower, tube_upper, color='#FDE68A', alpha=0.6, label='Epsilon Tube (ε = 0.1 Tolerance Band)')
plt.plot(x_grid_raw.ravel(), tube_upper, '--', color='#D97706', linewidth=1.5, label='Upper / Lower Bounds (±ε)')
plt.plot(x_grid_raw.ravel(), tube_lower, '--', color='#D97706', linewidth=1.5)

# 2. Plot SVR Prediction Line (Solid Blue)
plt.plot(x_grid_raw.ravel(), y_pred_line, '-', color='#1D4ED8', linewidth=3, label='SVR Prediction Curve (kernel=rbf, C=10.0)')

# 3. Plot Actual Egg Yolk Samples (647 Dots)
non_sv_mask = np.ones(len(X_raw), dtype=bool)
non_sv_mask[sv_indices] = False
plt.scatter(X_raw[non_sv_mask].ravel(), y[non_sv_mask], color='#EA580C', alpha=0.5, s=35, label='Egg Yolk Samples (N=647)')

# 4. Highlight Support Vectors (Green Ringed Dots)
plt.scatter(sv_x, sv_y, color='#10B981', edgecolors='#047857', s=70, linewidth=1.5, zorder=5, label=f'Support Vectors (N={len(sv_indices)})')

# Annotations & Formatting
plt.title('REAL PROJECT DATASET: SVR (RBF Kernel) Epsilon Tube & Support Vectors\nColor Redness Feature (a*) vs Yolk Color Fan Score', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Color Feature Redness (a*) from Image Dataset', fontsize=12, fontweight='bold')
plt.ylabel('Yolk Color Fan Score (Ground Truth 4 to 15)', fontsize=12, fontweight='bold')
plt.yticks(sorted(df['fan_score'].unique()))
plt.legend(fontsize=10, loc='upper left')
plt.tight_layout()

# Save plots
output_dev = r'c:\Users\Admin\Downloads\project_code\egg_yolk_project\model_dev\plots\real_svr_tube_visual.png'
output_brain = r'C:\Users\Admin\.gemini\antigravity\brain\baddf511-103e-4c9f-926d-8161adce50ce\real_svr_tube_visual.png'

plt.savefig(output_dev, dpi=300)
plt.savefig(output_brain, dpi=300)
plt.close()

print(f"Successfully generated real dataset SVR visualization!")
print(f"Saved to: {output_dev}")
print(f"Saved to: {output_brain}")
