import os
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.svm import SVR
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
feature_names_display = ['Red (R)', 'Green (G)', 'Blue (B)', 'Lightness (L*)', 'Red-Green (a*)', 'Yellow-Blue (b*)']

X = df[feature_cols].values
y = df['fan_score'].values

# Set style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Tahoma']

# 1. Feature Correlation Matrix Heatmap
plt.figure(figsize=(9, 7))
corr_df = df[feature_cols + ['fan_score']].copy()
corr_df.columns = feature_names_display + ['Yolk Fan Score']
corr_matrix = corr_df.corr()

sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5)
plt.title('Color Feature Correlation Matrix with Yolk Color Fan Score', fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
corr_path = os.path.join(plots_dir, 'feature_correlation_matrix.png')
plt.savefig(corr_path, dpi=300)
plt.close()
print(f"Saved: {corr_path}")

# 2. SVR Permutation Feature Importance Plot
pipeline = Pipeline([('scaler', StandardScaler()), ('regressor', SVR(kernel='rbf', C=10.0, epsilon=0.1))])
pipeline.fit(X, y)

result = permutation_importance(pipeline, X, y, n_repeats=30, random_state=42)
sorted_importances_idx = result.importances_mean.argsort()

plt.figure(figsize=(9, 6))
bars = plt.barh([feature_names_display[i] for i in sorted_importances_idx], result.importances_mean[sorted_importances_idx], color='#E1740A', edgecolor='black', alpha=0.85)

for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.4f}', ha='left', va='center', fontsize=10, fontweight='bold')

plt.title('Color Feature Importance in SVR Model (Permutation Importance)', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Importance Score (Decrease in Model Performance when Shuffled)', fontsize=12, fontweight='bold')
plt.xlim(0, max(result.importances_mean) * 1.15)
plt.tight_layout()
importance_path = os.path.join(plots_dir, 'feature_importance.png')
plt.savefig(importance_path, dpi=300)
plt.close()
print(f"Saved: {importance_path}")

# 3. Color Features Trend by Yolk Score (Line Plot across Classes 4 to 15)
class_means = df.groupby('fan_score')[feature_cols].mean()

plt.figure(figsize=(10, 6))
plt.plot(class_means.index, class_means['l'], 'o-', color='#3B82F6', linewidth=2.5, label='Lightness (L*)')
plt.plot(class_means.index, class_means['a'], 's-', color='#EF4444', linewidth=2.5, label='Redness (a*)')
plt.plot(class_means.index, class_means['b_lab'], '^-', color='#F59E0B', linewidth=2.5, label='Yellowness (b*)')

plt.title('CIELAB Color Features Trend across Yolk Color Fan Scores (4 to 15)', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Actual Yolk Color Fan Score Level', fontsize=12, fontweight='bold')
plt.ylabel('Average Color Value', fontsize=12, fontweight='bold')
plt.xticks(sorted(df['fan_score'].unique()))
plt.legend(fontsize=11)
plt.tight_layout()
trend_path = os.path.join(plots_dir, 'color_trends_by_score.png')
plt.savefig(trend_path, dpi=300)
plt.close()
print(f"Saved: {trend_path}")

print("All feature analysis plots generated successfully!")
