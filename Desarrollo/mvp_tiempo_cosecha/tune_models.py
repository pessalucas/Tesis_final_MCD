"""Shootout con hiperparametría: HistGBR vs XGBoost vs LightGBM.

Dataset con TODAS las features (base + clima). Búsqueda aleatoria de hiperparámetros
con GroupKFold por ID_COSECHA (sin fuga), optimizando MAE. Objetivo: ver cuánto se le
puede bajar el error al mejor modelo actual (HistGBR sin tunear ≈ 3.13-3.17 d).

Nota de entorno: XGBoost/LightGBM necesitan el runtime OpenMP libomp.dylib. No hay brew,
pero sklearn trae uno empaquetado; se enlazó al lib dir del Python de uv (ver notas del
proyecto). Los tres modelos manejan NaN nativamente.
"""
import warnings, time
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
from sklearn.model_selection import RandomizedSearchCV, GroupKFold
from sklearn.ensemble import HistGradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

RNG = 42
N_ITER = 30
L = pd.read_parquet("landmarks_clima.parquet")
FEATS = [c for c in L.columns if c not in ["ID_COSECHA", "ID_TANDA", "D"]]
X, y, groups = L[FEATS], L["D"].to_numpy(float), L["ID_COSECHA"].to_numpy()
print(f"Landmarks: {len(L)} | cosechas: {L['ID_COSECHA'].nunique()} | features: {len(FEATS)}")
cv = GroupKFold(n_splits=5)

MODELOS = {
    "HistGBR": (HistGradientBoostingRegressor(loss="absolute_error", random_state=RNG), {
        "learning_rate": [0.02, 0.03, 0.05, 0.08, 0.1],
        "max_iter": [200, 300, 400, 600],
        "max_leaf_nodes": [15, 31, 63],
        "max_depth": [3, 4, 5, 6, None],
        "min_samples_leaf": [20, 30, 40, 60, 80],
        "l2_regularization": [0, 0.5, 1, 2, 5],
    }),
    "XGBoost": (XGBRegressor(objective="reg:absoluteerror", tree_method="hist",
                             random_state=RNG, n_jobs=1, verbosity=0), {
        "n_estimators": [300, 500, 800],
        "learning_rate": [0.02, 0.03, 0.05, 0.08],
        "max_depth": [3, 4, 5, 6, 8],
        "min_child_weight": [1, 5, 10, 20],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "reg_lambda": [0.5, 1, 3, 5],
        "reg_alpha": [0, 0.5, 1],
    }),
    "LightGBM": (LGBMRegressor(objective="regression_l1", random_state=RNG,
                               n_jobs=1, verbose=-1, subsample_freq=1), {
        "n_estimators": [300, 500, 800],
        "learning_rate": [0.02, 0.03, 0.05, 0.08],
        "num_leaves": [15, 31, 63, 127],
        "max_depth": [-1, 4, 6, 8],
        "min_child_samples": [20, 40, 60, 100],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "reg_lambda": [0, 1, 3, 5],
    }),
}

resultados = {}
for nombre, (est, space) in MODELOS.items():
    t0 = time.time()
    print(f"\n>>> Tuning {nombre} ({N_ITER} configs x 5 folds)...")
    search = RandomizedSearchCV(est, space, n_iter=N_ITER, cv=cv,
                                scoring="neg_mean_absolute_error", n_jobs=-1,
                                random_state=RNG, refit=False)
    search.fit(X, y, groups=groups)
    best_i = search.best_index_
    mae = -search.best_score_
    std = search.cv_results_["std_test_score"][best_i]
    resultados[nombre] = (mae, std, search.best_params_)
    print(f"    MAE (GroupKFold) = {mae:.3f} ± {std:.3f} d   [{time.time()-t0:.0f}s]")
    print(f"    best params: {search.best_params_}")

print("\n" + "=" * 60)
print("RANKING (MAE en días, GroupKFold por cosecha, todas las features)")
print("=" * 60)
print(f"  {'baseline HistGBR sin tunear':32s} ~3.13-3.17")
for nombre, (mae, std, _) in sorted(resultados.items(), key=lambda kv: kv[1][0]):
    print(f"  {nombre:32s} {mae:.3f} ± {std:.3f}")
mejor = min(resultados.items(), key=lambda kv: kv[1][0])
print(f"\n  MEJOR: {mejor[0]} con MAE {mejor[1][0]:.3f} d")
