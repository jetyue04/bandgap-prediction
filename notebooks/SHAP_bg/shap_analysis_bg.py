#!/usr/bin/env python3
"""
SHAP Analysis for Bandgap Prediction
Based on Wang et al. (2025) 

Data: x_y_bandgap_normalized.csv (36 features, already normalized)
Model: XGBoost with hyperparameters from Wang et al.

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import xgboost as xgb
import shap
import warnings
warnings.filterwarnings('ignore')

# set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("=" * 80)
print("SHAP ANALYSIS FOR BANDGAP PREDICTION")
print("Using YOUR preprocessed data (36 features)")
print("=" * 80)

data = pd.read_csv('x_y_bandgap_normalized.csv')
print(f"Dataset loaded: {data.shape[0]} samples, {data.shape[1]} total columns")

# separate features and target (already normalized)
X = data.drop(columns=['band_gap'])
y = data['band_gap']

feature_names = X.columns.tolist()
print(f"\nFeatures: {len(feature_names)}")
print(f"Target: band_gap (range: {y.min():.3f} to {y.max():.3f} eV)")
print(f"\nNote: Data is already MinMax normalized and Pearson-filtered!")


X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.15, random_state=RANDOM_STATE
)

X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.15, random_state=RANDOM_STATE
)

print(f"Training set: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.2f}%)")
print(f"Validation set: {X_val.shape[0]} samples ({X_val.shape[0]/len(X)*100:.2f}%)")
print(f"Test set: {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.2f}%)")


print("\n[STEP 3] Training XGBoost model with YOUR hyperparameters...")

model = xgb.XGBRegressor(
    n_estimators=300,
    learning_rate=0.1,
    gamma=0,
    max_depth=6,
    subsample=1,
    colsample_bytree=0.8,
    reg_alpha=0,
    reg_lambda=1,
    objective='reg:squarederror',
    random_state=RANDOM_STATE,
    n_jobs=-1
)

model.fit(X_train, y_train)
print("Model training complete!")

y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)
y_test_pred = model.predict(X_test)

train_r2 = r2_score(y_train, y_train_pred)
train_mae = mean_absolute_error(y_train, y_train_pred)
val_r2 = r2_score(y_val, y_val_pred)
val_mae = mean_absolute_error(y_val, y_val_pred)
test_r2 = r2_score(y_test, y_test_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

print(f"\nModel Performance:")
print(f"  Training   - R²: {train_r2:.4f}, MAE: {train_mae:.4f} eV")
print(f"  Validation - R²: {val_r2:.4f}, MAE: {val_mae:.4f} eV")
print(f"  Test       - R²: {test_r2:.4f}, MAE: {test_mae:.4f} eV, RMSE: {test_rmse:.4f} eV")

# SHAP Analysis
# SHAP explainer
explainer = shap.TreeExplainer(model)

# SHAP values for test set
shap_values = explainer.shap_values(X_test)

# SHAP Explanation object
shap_explanation = shap.Explanation(
    values=shap_values,
    base_values=explainer.expected_value,
    data=X_test.values,
    feature_names=feature_names
)

print(f"SHAP values calculated for {len(shap_values)} test samples")
print(f"Expected value (base prediction): {explainer.expected_value:.4f} eV")

# Feature Importance Bar Plot
print("  5.1 Creating feature importance bar plot...")
plt.figure(figsize=(12, 10))
shap.summary_plot(shap_values, X_test, plot_type="bar",
                 feature_names=feature_names, show=False, max_display=20)
plt.title("SHAP Feature Importance - Bandgap Prediction\n(Your 36 Selected Features)",
         fontsize=14, fontweight='bold', pad=20)
plt.xlabel("Mean |SHAP value|", fontsize=12)
plt.tight_layout()
plt.savefig('shap_bandgap_importance_bar.png', dpi=300, bbox_inches='tight')
print("     Saved: shap_bandgap_importance_bar.png")
plt.close()

# SHAP Summary Plot (Beeswarm)
plt.figure(figsize=(12, 10))
shap.summary_plot(shap_values, X_test,
                 feature_names=feature_names, show=False, max_display=20)
plt.title("SHAP Summary Plot - Bandgap Prediction\n(Impact & Direction of Features)",
         fontsize=14, fontweight='bold', pad=20)
plt.xlabel("SHAP value (impact on bandgap prediction)", fontsize=12)
plt.tight_layout()
plt.savefig('shap_bandgap_summary_beeswarm.png', dpi=300, bbox_inches='tight')
print("     Saved: shap_bandgap_summary_beeswarm.png")
plt.close()

# SHAP Dependence Plots
feature_importance = np.abs(shap_values).mean(0)
top_3_indices = np.argsort(feature_importance)[-3:][::-1]
top_3_features = [feature_names[i] for i in top_3_indices]

print(f"     Top 3 features: {', '.join(top_3_features)}")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for idx, (feature_idx, feature) in enumerate(zip(top_3_indices, top_3_features)):
    plt.sca(axes[idx])
    shap.dependence_plot(
        feature_idx,
        shap_values,
        X_test,
        feature_names=feature_names,
        show=False
    )
    axes[idx].set_title(f"{feature}\n(Rank #{idx+1})",
                       fontsize=11, fontweight='bold')

plt.suptitle("SHAP Dependence Plots - Top 3 Most Important Features",
            fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('shap_bandgap_dependence_top3.png', dpi=300, bbox_inches='tight')
print("     Saved: shap_bandgap_dependence_top3.png")
plt.close()

# SHAP Waterfall Plot 
sample_idx = 42  # Example instance
plt.figure(figsize=(10, 10))
shap.waterfall_plot(shap_explanation[sample_idx], max_display=20, show=False)
plt.title(f"SHAP Waterfall Plot for Sample #{sample_idx}\n" +
         f"Predicted: {y_test_pred[sample_idx]:.3f} eV, Actual: {y_test.iloc[sample_idx]:.3f} eV",
         fontsize=12, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('shap_bandgap_waterfall_sample.png', dpi=300, bbox_inches='tight')
print("     Saved: shap_bandgap_waterfall_sample.png")
plt.close()

# SHAP Force Plot
shap.initjs()
force_plot = shap.force_plot(
    explainer.expected_value,
    shap_values[sample_idx],
    X_test.iloc[sample_idx],
    feature_names=feature_names
)
shap.save_html('shap_bandgap_force_plot.html', force_plot)
print("     Saved: shap_bandgap_force_plot.html")

# mean absolute SHAP values
mean_abs_shap = np.abs(shap_values).mean(0)
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Mean_|SHAP|': mean_abs_shap
}).sort_values('Mean_|SHAP|', ascending=False)

print("\nTop 15 Most Important Features (by mean |SHAP|):")
print(feature_importance_df.head(15).to_string(index=False))

# correlation direction
print("\n\nFeature Effect Analysis:")
print("-" * 80)
print(f"{'Feature':<40s} {'Effect':<10s} {'Direction'}")
print("-" * 80)

for feature in feature_importance_df.head(15)['Feature']:
    feature_idx = feature_names.index(feature)
    mean_shap = shap_values[:, feature_idx].mean()
    effect = "POSITIVE" if mean_shap > 0 else "NEGATIVE"
    arrow = "↑ increases bandgap" if mean_shap > 0 else "↓ decreases bandgap"
    print(f"{feature:<40s} {effect:<10s} {arrow}")

# Save feature importance
feature_importance_df.to_csv('shap_bandgap_feature_importance.csv', index=False)
print("Saved: shap_bandgap_feature_importance.csv")

# Save SHAP values
shap_df = pd.DataFrame(shap_values, columns=feature_names)
shap_df.to_csv('shap_bandgap_values_test_set.csv', index=False)
print("Saved: shap_bandgap_values_test_set.csv")

# Save model predictions
predictions_df = pd.DataFrame({
    'Actual_Bandgap': y_test.values,
    'Predicted_Bandgap': y_test_pred,
    'Absolute_Error': np.abs(y_test.values - y_test_pred)
})
predictions_df.to_csv('shap_bandgap_predictions_test_set.csv', index=False)
print("Saved: shap_bandgap_predictions_test_set.csv")

# Save model
# model.save_model('xgboost_bandgap_model.json')
# print("Saved: xgboost_bandgap_model.json")

# Save detailed report
with open('shap_bandgap_analysis_report.txt', 'w') as f:
    f.write("-" * 80 + "\n")
    f.write("SHAP Analysis Report - Bandgap Prediction\n")
    f.write("-" * 80 + "\n\n")

    f.write("DATASET INFO\n")
    f.write("-" * 80 + "\n")
    f.write(f"Total samples: {len(data)}\n")
    f.write(f"Features: {len(feature_names)} (after Pearson+mRMR selection)\n")
    f.write(f"Training: {len(X_train)} samples\n")
    f.write(f"Validation: {len(X_val)} samples\n")
    f.write(f"Test: {len(X_test)} samples\n\n")

    f.write("MODEL PERFORMANCE\n")
    f.write("-" * 80 + "\n")
    f.write(f"Training   - R²: {train_r2:.4f}, MAE: {train_mae:.4f} eV\n")
    f.write(f"Validation - R²: {val_r2:.4f}, MAE: {val_mae:.4f} eV\n")
    f.write(f"Test       - R²: {test_r2:.4f}, MAE: {test_mae:.4f} eV, RMSE: {test_rmse:.4f} eV\n\n")

    f.write("FEATURE IMPORTANCE (Mean |SHAP| values)\n")
    f.write("-" * 80 + "\n")
    f.write(feature_importance_df.to_string(index=False))
    f.write("\n\n")

    f.write("FEATURE EFFECTS ON BANDGAP\n")
    f.write("-" * 80 + "\n")
    for feature in feature_names:
        feature_idx = feature_names.index(feature)
        mean_shap = shap_values[:, feature_idx].mean()
        effect = "POSITIVE" if mean_shap > 0 else "NEGATIVE"
        f.write(f"{feature:<40s} {effect:<10s} (mean SHAP: {mean_shap:+.6f})\n")

print("saved: shap_bandgap_analysis_report.txt")

print("\nGenerated Files:")
print("  1. shap_bandgap_importance_bar.png")
print("  2. shap_bandgap_summary_beeswarm.png")
print("  3. shap_bandgap_dependence_top3.png")
print("  4. shap_bandgap_waterfall_sample.png")
print("  5. shap_bandgap_force_plot.html")
print("  6. shap_bandgap_feature_importance.csv")
print("  7. shap_bandgap_values_test_set.csv")
print("  8. shap_bandgap_predictions_test_set.csv")
print("  9. xgboost_bandgap_model.json")
print(" 10. shap_bandgap_analysis_report.txt")
print("\nAll files saved to current directory.")
