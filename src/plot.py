y_pred_formation_train = bst_formation.predict(X_formation_train, num_iteration=bst_formation.best_iteration)
y_pred_formation_val = bst_formation.predict(X_formation_val, num_iteration=bst_formation.best_iteration)
fig, ax = plt.subplots(figsize=(7, 6))

# Training set
ax.scatter(y_formation_train, y_pred_formation_train, 
           label='Train', alpha=0.4, color='blue')

# Validation set
ax.scatter(y_formation_val, y_pred_formation_val, 
           label='Validation', alpha=0.8, color='orange')

# Perfect prediction line
low = min(min(y_formation_train), min(y_formation_val))
high = max(max(y_formation_train), max(y_formation_val))
ax.plot([low, high], [low, high], '--k', color = 'red', label='Perfect prediction')

# Labels
ax.set_xlabel("True Band Gap")
ax.set_ylabel("Predicted Band Gap")
ax.set_title("LightGBM Predictions vs True Values")
ax.legend(loc='lower right')
ax.grid(True)

# Metrics
rmse = np.sqrt(mean_squared_error(y_formation_val, y_pred_formation_val))
r2 = r2_score(y_formation_val, y_pred_formation_val)
mae = mean_absolute_error(y_formation_val, y_pred_formation_val)

textstr = (
    f'Validation R²: {r2:.4f}\n'
    f'Validation MAE: {mae:.4f} eV\n'
    f'Validation RMSE: {rmse:.4f} eV'
)

props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
        fontsize=11, verticalalignment='top',
        bbox=props, family='monospace')

plt.tight_layout()
plt.show()