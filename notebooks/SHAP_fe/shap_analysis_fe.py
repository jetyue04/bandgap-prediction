#!/usr/bin/env python3
"""
SHAP Analysis for Formation Energy Prediction in Lead-Free Double Perovskites
Based on Wang et al. (2025)

Data: x_y_formation_normalized.csv (37 features, already normalized)
Model: XGBoost with hyperparameters from Wang et al.

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import xgboost as xgb
import shap
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("-" * 80)
print("SHAP Analysis for Formation Energy Prediction")
print("based on Wang et al. (2025) - Molecules")
print("-" * 80)

# Load data
df = pd.read_excel('dataset_1053.xlsx')
print(f"Dataset loaded: {df.shape[0]} samples")

# Feature selection for formation energy (18 features from Wang et al.)
selected_features = [
    'B1_ionic_radius',              # RB'
    'A_electronegativity',          # χA
    'B1_electronegativity',         # χB'
    'B2_electronegativity',         # χB'' 
    'B1_electron_affinity',         # EAB'
    'B2_electron_affinity',         # EAB''
    'X_electron_affinity',          # EAX
    'B1_first_ionization_energy',   # IE1B'
    'X_first_ionization_energy',    # IE1X 
    'A_second_ionization_energy',   # IE2A
    'B1_second_ionization_energy',  # IE2B'
    'A_melting_point',              # TmA
    'B1_melting_point',             # TmB'
    'B1_thermal_conductivity',      # kB'
    'A_boiling_point',              # TbA
    'B1_boiling_point',             # TbB'
    'X_boiling_point',              # TbX
    'X_third_ionization_energy'     # IE3X
]

X = df[selected_features].copy()
y = df['formation_energy'].copy()

print(f"Selected {len(selected_features)} features for formation energy prediction")
print(f"Target range: {y.min():.3f} to {y.max():.3f} eV/atom")

# Normalize
scaler = MinMaxScaler()
X_scaled = pd.DataFrame(
    scaler.fit_transform(X),
    columns=selected_features,
    index=X.index
)

# Split data
X_temp, X_test, y_temp, y_test = train_test_split(
    X_scaled, y, test_size=0.15, random_state=RANDOM_STATE
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.15, random_state=RANDOM_STATE
)

print(f"Training: {X_train.shape[0]}, Validation: {X_val.shape[0]}, Test: {X_test.shape[0]}")

# Train XGBoost
xgb_params = {
    'learning_rate': 0.1,
    'gamma': 0,
    'colsample_bytree': 0.6,
    'max_depth': 3,
    'n_estimators': 300,
    'reg_alpha': 0,
    'reg_lambda': 1,
    'subsample': 1.0,
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}

model = xgb.XGBRegressor(**xgb_params)
model.fit(X_train, y_train)

# Evaluate
y_test_pred = model.predict(X_test)
test_r2 = r2_score(y_test, y_test_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

print(f"\nModel Performance:")
print(f"  Test R²:   {test_r2:.4f}")
print(f"  Test MAE:  {test_mae:.4f} eV/atom")
print(f"  Test RMSE: {test_rmse:.4f} eV/atom")
print(f"  (Wang et al.: R²=0.959, MAE=0.013, RMSE=0.091)")

# SHAP Analysis
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap_explanation = shap.Explanation(
    values=shap_values,
    base_values=explainer.expected_value,
    data=X_test.values,
    feature_names=selected_features
)

# Visualizations

# Feature importance
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test, plot_type="bar",
                 feature_names=selected_features, show=False, max_display=18)
plt.title("SHAP Feature Importance for Formation Energy", 
         fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('shap_formation_energy_importance.png', dpi=300, bbox_inches='tight')
print("  Saved: shap_formation_energy_importance.png")
plt.close()

# Summary plot
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test,
                 feature_names=selected_features, show=False, max_display=18)
plt.title("SHAP Summary for Formation Energy", 
         fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('shap_formation_energy_summary.png', dpi=300, bbox_inches='tight')
print("  Saved: shap_formation_energy_summary.png")
plt.close()

# Top 3 dependence plots
feature_importance = np.abs(shap_values).mean(0)
top_3_indices = np.argsort(feature_importance)[-3:][::-1]
top_3_features = [selected_features[i] for i in top_3_indices]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for idx, (feature_idx, feature) in enumerate(zip(top_3_indices, top_3_features)):
    plt.sca(axes[idx])
    shap.dependence_plot(feature_idx, shap_values, X_test,
                        feature_names=selected_features, show=False)
    axes[idx].set_title(f"{feature}\n(Rank #{idx+1})", 
                       fontsize=11, fontweight='bold')
plt.suptitle("SHAP Dependence Plots - Formation Energy", 
            fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('shap_formation_dependence_top3.png', dpi=300, bbox_inches='tight')
print("  Saved: shap_formation_dependence_top3.png")
plt.close()

# Save results
feature_importance_df = pd.DataFrame({
    'Feature': selected_features,
    'Mean_|SHAP|': np.abs(shap_values).mean(0)
}).sort_values('Mean_|SHAP|', ascending=False)
feature_importance_df.to_csv('shap_formation_importance.csv', index=False)
print("  Saved: shap_formation_importance.csv")

pd.DataFrame(shap_values, columns=selected_features).to_csv(
    'shap_formation_values.csv', index=False
)
print("  Saved: shap_formation_values.csv")

# Report
print("\n" + "-" * 80)
print("Key Findings:")
print("-" * 80)
print("\nMost Important Features:")
for idx, row in feature_importance_df.head(5).iterrows():
    feature = row['Feature']
    feature_idx = selected_features.index(feature)
    mean_shap = shap_values[:, feature_idx].mean()
    effect = "+" if mean_shap > 0 else "-"
    print(f"  {row['Feature']:40s} {effect}")

print("\n\nPhysical Interpretation (Wang et al.):")
print("-" * 80)
print("  • X_first_ionization_energy (IE1X):  NEGATIVE effect")
print("     Higher IE1X means more stable → more negative formation energy")
print("\n  • B'_electronegativity (χB'):         POSITIVE effect")
print("  • B''_electronegativity (χB''):        POSITIVE effect")
print("     Higher electronegativity → less negative formation energy")