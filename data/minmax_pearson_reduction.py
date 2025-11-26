"""
MinMax Scaling and Pearson-based Dimension Reduction
For Double Perovskite Bandgap Prediction Project
Based on: Wang et al. (2025) - Molecules, Section 2.1 and 2.2

"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import mutual_info_regression


def minmax_scale_features(df, feature_columns):
    X = df[feature_columns].copy()
    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(X)
    X_normalized = pd.DataFrame(X_normalized, columns=feature_columns, index=X.index)
    
    print(f"MinMax scaling completed")
    print(f"  Features scaled: {len(feature_columns)}")
    print(f"  Value range: [{X_normalized.min().min():.6f}, {X_normalized.max().max():.6f}]")
    
    return X_normalized, scaler


def pearson_correlation_selection(X_normalized, y_target, correlation_threshold=0.90):
    print(f"\n--- Pearson Correlation Feature Selection ---")
    print(f"Initial features: {len(X_normalized.columns)}")
    print(f"Correlation threshold: {correlation_threshold}")
    
    mi_scores = mutual_info_regression(X_normalized, y_target, random_state=42)
    mrmr_ranking = pd.Series(mi_scores, index=X_normalized.columns).sort_values(ascending=False)
    
    feature_corr_matrix = X_normalized.corr()
    
    selected_features = set(X_normalized.columns)
    removed_features = []
    
    for i in range(len(feature_corr_matrix.columns)):
        for j in range(i+1, len(feature_corr_matrix.columns)):
            feat_i = feature_corr_matrix.columns[i]
            feat_j = feature_corr_matrix.columns[j]
            
            if feat_i in selected_features and feat_j in selected_features:
                corr_val = abs(feature_corr_matrix.iloc[i, j])
                
                if corr_val > correlation_threshold:
                    if mrmr_ranking[feat_i] < mrmr_ranking[feat_j]:
                        selected_features.remove(feat_i)
                        removed_features.append((feat_i, feat_j, corr_val))
                    else:
                        selected_features.remove(feat_j)
                        removed_features.append((feat_j, feat_i, corr_val))
    
    selected_features = sorted(list(selected_features))
    
    print(f"Removed features: {len(removed_features)}")
    print(f"Final features: {len(selected_features)}")
    
    return selected_features, removed_features


def process_perovskite_data(input_file, correlation_threshold=0.90):
    print(f"\nLoading: {input_file}")
    df = pd.read_excel(input_file)
    print(f"Dataset shape: {df.shape}")
    
    identifiers = ['A2BBX6', 'A', 'B1', 'B2', 'X']
    targets = ['band_gap', 'formation_energy']
    feature_columns = [col for col in df.columns if col not in identifiers + targets]
    
    print(f"Features: {len(feature_columns)}")
    print(f"Targets: {targets}")
    
    y_bandgap = df['band_gap'].copy()
    y_formation = df['formation_energy'].copy()

    print("="*80)
    print("1: MinMax Scaling")
    print("="*80)
    
    X_normalized, scaler = minmax_scale_features(df, feature_columns)
    print("\n" + "="*80)
    print("2: Feature Selection for Bandgap Prediction")
    print("="*80)
    
    bandgap_features, bandgap_removed = pearson_correlation_selection(
        X_normalized, y_bandgap, correlation_threshold
    )
    
    print("\n" + "="*80)
    print("3: Feature Selection for Formation Energy Prediction")
    print("="*80)
    
    formation_features, formation_removed = pearson_correlation_selection(
        X_normalized, y_formation, correlation_threshold
    )
    
    print("\n" + "="*80)
    print("4: Output Datasets")
    print("="*80)
    
    X_bandgap = X_normalized[bandgap_features].copy()
    X_bandgap['band_gap'] = y_bandgap
    
    X_formation = X_normalized[formation_features].copy()
    X_formation['formation_energy'] = y_formation
    
    bandgap_full = pd.concat([
        df[identifiers],
        X_normalized[bandgap_features],
        df[['band_gap']]
    ], axis=1)
    
    formation_full = pd.concat([
        df[identifiers],
        X_normalized[formation_features],
        df[['formation_energy']]
    ], axis=1)
    
    print(f"Bandgap dataset: {X_bandgap.shape}")
    print(f"Formation energy dataset: {X_formation.shape}")
    
    print("\n" + "="*80)
    print("5: Saving Results")
    print("="*80)
    
    X_bandgap.to_csv('x_y_bandgap_normalized.csv', index=False)
    print(f"Saved: x_y_bandgap_normalized.csv")
    
    X_formation.to_csv('x_y_formation_normalized.csv', index=False)
    print(f"Saved: x_y_formation_normalized.csv")
    
    bandgap_full.to_csv('bandgap_dataset_processed.csv', index=False)
    print(f"Saved: bandgap_dataset_processed.csv")
    
    formation_full.to_csv('formation_energy_dataset_processed.csv', index=False)
    print(f"Saved: formation_energy_dataset_processed.csv")
    
    with open('selected_features.txt', 'w') as f:
        f.write("BANDGAP PREDICTION - SELECTED FEATURES\n")
        f.write("="*50 + "\n")
        f.write(f"Total: {len(bandgap_features)} features\n\n")
        for i, feat in enumerate(bandgap_features, 1):
            f.write(f"{i:2d}. {feat}\n")
        
        f.write("\n\n")
        f.write("FORMATION ENERGY PREDICTION - SELECTED FEATURES\n")
        f.write("="*50 + "\n")
        f.write(f"Total: {len(formation_features)} features\n\n")
        for i, feat in enumerate(formation_features, 1):
            f.write(f"{i:2d}. {feat}\n")
        
        f.write("\n\n")
        f.write("REMOVED FEATURES (Bandgap)\n")
        f.write("="*50 + "\n")
        for removed, kept, corr in bandgap_removed:
            f.write(f"  {removed} (corr={corr:.3f} with {kept})\n")
        
        f.write("\n\n")
        f.write("REMOVED FEATURES (Formation Energy)\n")
        f.write("="*50 + "\n")
        for removed, kept, corr in formation_removed:
            f.write(f"  {removed} (corr={corr:.3f} with {kept})\n")
    
    print(f"Saved: selected_features.txt")
    
    # Summary
    print("\n" + "="*80)
    print("Prossing complege")
    print("="*80)
    print(f"\nSummary:")
    print(f"  Original features: {len(feature_columns)}")
    print(f"  Bandgap features: {len(bandgap_features)} (removed {len(bandgap_removed)})")
    print(f"  Formation energy features: {len(formation_features)} (removed {len(formation_removed)})")
    print(f"  Correlation threshold: {correlation_threshold}")
    
    return {
        'X_normalized': X_normalized,
        'scaler': scaler,
        'bandgap_features': bandgap_features,
        'formation_features': formation_features,
        'bandgap_removed': bandgap_removed,
        'formation_removed': formation_removed,
        'X_bandgap': X_bandgap,
        'X_formation': X_formation
    }


if __name__ == "__main__":
    results = process_perovskite_data(
        input_file='dataset_1053.xlsx',
        correlation_threshold=0.90
    )