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

feature_cols = ['r', 'g', 'b', 'l', 'a', 'b_lab']
X = df[feature_cols].values
y = df['fan_score'].values

# Fit full SVR Pipeline
pipeline = Pipeline([('scaler', StandardScaler()), ('regressor', SVR(kernel='rbf', C=10.0, epsilon=0.1))])
pipeline.fit(X, y)

# Compute continuous predictions
y_pred = pipeline.predict(X)
errors = np.abs(y - y_pred)

# Count inside tube vs outside tube (epsilon = 0.1)
epsilon = 0.1
inside_mask = errors <= epsilon
outside_mask = errors > epsilon

inside_count = int(np.sum(inside_mask))
outside_count = int(np.sum(outside_mask))
total_count = len(y)

inside_pct = (inside_count / total_count) * 100.0
outside_pct = (outside_count / total_count) * 100.0

print(f"Total Samples: {total_count}")
print(f"Inside Epsilon-Tube (<= 0.1): {inside_count} ({inside_pct:.1f}%)")
print(f"Outside Epsilon-Tube (> 0.1): {outside_count} ({outside_pct:.1f}%)")

# Create 2-panel Detailed Zoomed Visual Figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={'width_ratios': [1.2, 1]})
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# --- Panel 1: Zoomed-in Tube Visualization (Sample Subset of 100 points for extreme clarity) ---
sample_indices = np.random.choice(total_count, min(total_count, 120), replace=False)
sample_indices.sort()

y_sub = y[sample_indices]
pred_sub = y_pred[sample_indices]
err_sub = errors[sample_indices]
ins_sub = err_sub <= epsilon
out_sub = err_sub > epsilon

x_axis = np.arange(len(sample_indices))

# Draw Epsilon Tube Band (Translucent Amber Shading)
ax1.fill_between(x_axis, pred_sub - epsilon, pred_sub + epsilon, color='#FDE68A', alpha=0.7, label=f'Epsilon Tube (ε = 0.1 Tolerance Band)')
ax1.plot(x_axis, pred_sub + epsilon, '--', color='#D97706', linewidth=1.5, label='Upper / Lower Bounds (±0.1)')
ax1.plot(x_axis, pred_sub - epsilon, '--', color='#D97706', linewidth=1.5)

# Draw SVR Prediction Curve Line (Blue)
ax1.plot(x_axis, pred_sub, '-', color='#1D4ED8', linewidth=2.5, label='SVR Prediction Curve f(x)')

# Plot Inside Points (Green Dots)
ax1.scatter(x_axis[ins_sub], y_sub[ins_sub], color='#10B981', s=50, edgecolors='black', linewidth=0.5, zorder=5, label=f'Points Inside Tube (Compliant)')

# Plot Outside Points (Red Dots with Error Stems)
ax1.scatter(x_axis[out_sub], y_sub[out_sub], color='#EF4444', s=60, edgecolors='black', linewidth=0.8, zorder=6, label=f'Points Outside Tube (Penalized)')

# Draw vertical error stems for outside points
for idx in np.where(out_sub)[0]:
    ax1.plot([idx, idx], [pred_sub[idx], y_sub[idx]], 'r:', linewidth=1.2)

ax1.set_title('ZOOMED-IN VIEW OF EPSILON-TUBE (SVR RBF Kernel)\nShowing Individual Egg Yolk Samples & Prediction Errors', fontsize=13, fontweight='bold', pad=12)
ax1.set_xlabel('Sample Index (Subsample View N=120)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Yolk Color Fan Score', fontsize=11, fontweight='bold')
ax1.legend(fontsize=9, loc='upper left')

# --- Panel 2: Total Count Summary Pie & Bar Breakdown ---
colors_pie = ['#10B981', '#EF4444']
labels_pie = [f'Inside Tube (<= 0.1)\n{inside_count} samples ({inside_pct:.1f}%)', f'Outside Tube (> 0.1)\n{outside_count} samples ({outside_pct:.1f}%)']
explode = (0.05, 0)

wedges, texts, autotexts = ax2.pie(
    [inside_count, outside_count],
    labels=labels_pie,
    autopct='%1.1f%%',
    startangle=140,
    colors=colors_pie,
    explode=explode,
    shadow=True,
    textprops=dict(fontsize=11, fontweight='bold')
)

plt.setp(autotexts, size=12, weight="bold", color="white")
ax2.set_title(f'TOTAL DATASET BREAKDOWN (N={total_count} Egg Samples)\nDistribution of Samples Inside vs Outside Epsilon-Tube', fontsize=13, fontweight='bold', pad=12)

plt.tight_layout()

# Save plots
out_dev = r'c:\Users\Admin\Downloads\project_code\egg_yolk_project\model_dev\plots\detailed_tube_count_breakdown.png'
out_brain = r'C:\Users\Admin\.gemini\antigravity\brain\baddf511-103e-4c9f-926d-8161adce50ce\detailed_tube_count_breakdown.png'

plt.savefig(out_dev, dpi=300)
plt.savefig(out_brain, dpi=300)
plt.close()

print(f"Successfully saved detailed tube count breakdown visual to:\n{out_dev}\n{out_brain}")
