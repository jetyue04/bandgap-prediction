import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
# from scipy.stats import pearsonr

df = pd.read_csv('../data/x_y_bandgap_normalized.csv')
X = df.drop(columns=['band_gap'])
y = df['band_gap']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

param_grid = {
    "num_leaves": [31, 64, 127],
    "max_depth": [-1, 6, 8],
    "learning_rate": [0.1, 0.05, 0.01],
    "n_estimators": [500, 1000, 2000],
    "subsample": [0.8, 0.9],
    "colsample_bytree": [0.8, 0.9],
    "reg_lambda": [0.0, 1.0, 5.0],   # L2 regularization
}
model = LGBMRegressor(objective="regression", boosting_type="gbdt")

grid = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring="neg_root_mean_squared_error",  # <--- RMSE metric
    cv=3,
    verbose=1,
    n_jobs=-1
)

grid.fit(X_train, y_train)

print("Best params:", grid.best_params_)
print("Best validation RMSE:", -grid.best_score_)

