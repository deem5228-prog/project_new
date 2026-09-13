# 🥚 Egg Yolk Color Prediction - Model Development (SVR)

This repository contains the complete **Model Development Lifecycle** for predicting egg yolk color scores (DSM Yolk Color Fan 1–15) using Digital Image Processing, Center Circular Masking, and Support Vector Regression (SVR).

---

## 📁 Repository Structure

```
├── color_service.py           # Core color extraction (Center Circular Masking R=42% & CIELAB conversion)
├── build_features.py          # Batch feature extraction pipeline (generates features.csv & labels.csv)
├── train_compare_models.py    # Stratified 5-Fold Cross-Validation & Benchmark of 5 ML Models
├── train_compare_models.ipynb # Jupyter Notebook with evaluation tables & per-class error breakdown
├── export_best_model.py       # Full-dataset training & production export script
├── model.pkl                  # Serialized SVR (RBF Kernel) pipeline model (40.5 KB)
├── requirements.txt           # Required Python packages
└── data/                      # Dataset tables (features.csv, labels.csv)
```

---

## 📊 Benchmark Evaluation Results (Stratified 5-Fold Cross-Validation)

```
==========================================================================================
MODEL TRAIN vs TEST COMPARISON RESULTS (Stratified 5-Fold Cross-Validation)
==========================================================================================
            Model  Train R2  Test R2  Train MAE  Test MAE  Train RMSE  Test RMSE
 SVR (RBF Kernel)    0.9295   0.9175     0.5814    0.6453      0.7915     0.8532  🏆
    Random Forest    0.9871   0.9087     0.2462    0.6485      0.3389     0.8923
Gradient Boosting    0.9663   0.9063     0.4209    0.6621      0.5473     0.9038
Linear Regression    0.8749   0.8718     0.8074    0.8136      1.0542     1.0645
 Ridge Regression    0.8574   0.8545     0.8775    0.8835      1.1256     1.1355
==========================================================================================

Best Performing Model: SVR (RBF Kernel) (Test R^2 = 0.9175, Test MAE = 0.6453, Test RMSE = 0.8532)
```

---

## 📋 Per-Class Regression Error Breakdown for SVR (RBF Kernel)

```
==========================================================================================
=== PER-CLASS REGRESSION ERROR BREAKDOWN FOR TOP MODEL (SVR (RBF Kernel)) ===
==========================================================================================
           Model  Class (Fan Score)  Samples  Mean Pred    MAE     RMSE
SVR (RBF Kernel)                  4       36       4.29   0.3437  0.6434
SVR (RBF Kernel)                  5       41       5.48   0.7397  0.9146
SVR (RBF Kernel)                  6       49       6.14   0.6864  0.8799
SVR (RBF Kernel)                  7       42       7.29   0.7035  0.9421
SVR (RBF Kernel)                  8       64       8.26   0.5288  0.6512
SVR (RBF Kernel)                  9      105       9.06   0.6079  0.7664
SVR (RBF Kernel)                 10       99      10.02   0.6547  0.8258
SVR (RBF Kernel)                 11       75      10.77   0.6839  0.9368
SVR (RBF Kernel)                 12       33      11.68   0.8708  1.1319
SVR (RBF Kernel)                 13       21      12.66   0.9676  1.0728
SVR (RBF Kernel)                 14       33      13.47   0.7329  0.9236
SVR (RBF Kernel)                 15       49      14.61   0.5019  0.8145
==========================================================================================
```

---

## 🚀 How to Run

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Extract Features (Center Circular Masking):**
   ```bash
   python build_features.py
   ```

3. **Train & Evaluate Models (5-Fold CV):**
   ```bash
   python train_compare_models.py
   ```

4. **Export Best Model:**
   ```bash
   python export_best_model.py
   ```
