"""
Plot Updated Real SVR Epsilon Tube Visualization with Clean Masked Features (N=567 Support Vectors)
"""

import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11

# Paths
base_dir = os.path.dirname(__file__)
data_csv = os.path.join(base_dir, 'data', 'features.csv')
plots_dir = os.path.join(base_dir, 'plots')
brain_dir = r'C:\Users\Admin\.gemini\antigravity\brain\baddf511-103e-4c9f-926d-8161adce50ce'

df = pd.read_csv(data_csv)

# 1D SVR on Redness Feature a* to visualize the exact Epsilon Tube
X_a = df[['a']].values
y = df['fan_score'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_a)

svr = SVR(kernel='rbf', C=10.0, epsilon=0.1)
svr.fit(X_scaled, y)

# Smooth curve points
x_plot = np.linspace(X_a.min() - 2, X_a.max() + 2, 500).reshape(-1, 1)
x_plot_scaled = scaler.transform(x_plot)
y_pred_curve = svr.predict(x_plot_scaled)

# Support Vectors indices
sv_indices = svr.support_
sv_x = X_a[sv_indices].flatten()
sv_y = y[sv_indices]

# Plotting
plt.figure(figsize=(12, 7.5))

# Plot all samples
plt.scatter(df['a'], y, color='#F97316', alpha=0.45, s=35, label=f'Egg Yolk Samples (N={len(df)})')

# Plot Support Vectors
plt.scatter(sv_x, sv_y, s=75, facecolors='#10B981', edgecolors='#064E3B', linewidths=1.2,
            zorder=4, label=f'Support Vectors (N={len(sv_indices)})')

# Plot Epsilon Tube (Tolerated Error Band)
eps = 0.1
plt.fill_between(x_plot.flatten(), y_pred_curve - eps, y_pred_curve + eps,
                 color='#FDE68A', alpha=0.7, label=r'Epsilon Tube ($\epsilon = 0.1$ Tolerance Band)', zorder=2)
plt.plot(x_plot.flatten(), y_pred_curve + eps, '--', color='#D97706', linewidth=1.5, label=r'Upper / Lower Bounds ($\pm\epsilon$)', zorder=3)
plt.plot(x_plot.flatten(), y_pred_curve - eps, '--', color='#D97706', linewidth=1.5, zorder=3)

# Plot SVR Regression Curve
plt.plot(x_plot.flatten(), y_pred_curve, color='#1D4ED8', linewidth=3.2,
         label='SVR Prediction Curve (kernel=rbf, C=10.0)', zorder=5)

plt.title('REAL PROJECT DATASET: SVR (RBF Kernel) Epsilon Tube & Support Vectors\nColor Redness Feature (a*) vs Yolk Color Fan Score',
          fontsize=13, fontweight='bold', pad=14)
plt.xlabel('Color Feature: Redness Axis a* (CIELAB Space)', fontsize=11, fontweight='bold')
plt.ylabel('Yolk Color Fan Score (Classes 4–15)', fontsize=11, fontweight='bold')
plt.yticks(range(4, 16))
plt.legend(loc='upper left', frameon=True, fontsize=10)
plt.tight_layout()

tube_path = os.path.join(plots_dir, 'real_svr_tube_visual.png')
plt.savefig(tube_path, dpi=300, bbox_inches='tight')
plt.close()

dest_brain = os.path.join(brain_dir, 'real_svr_tube_visual.png')
shutil.copy2(tube_path, dest_brain)

print("Generated updated SVR tube visual successfully!")
