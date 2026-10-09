"""Challenge de modelos — ¿qué tecnología predice mejor el tiempo de cosecha?

Diseño de evaluación:
  - TEST TEMPORAL: cosechas iniciadas en el ÚLTIMO AÑO (>= 2025-09-26) = realismo de
    producción (entrenar con el pasado, predecir el futuro). 42 cosechas de test.
  - HPO: RandomizedSearchCV (n_iter=50) con GroupKFold(5) por ID_COSECHA SOLO sobre el
    train (loop interno). Nunca toca el test. Scoring = MAE (coherente con el objetivo).
  - 7 tecnologías de 5 familias: lineal (ElasticNet), instancias (kNN), bagging
    (RandomForest), boosting (LightGBM, XGBoost, CatBoost), red neuronal (MLP).
  - Preprocesado DENTRO del fold (Pipeline: imputación + escalado donde hace falta) para
    no filtrar información. Los árboles de boosting manejan NaN nativo.
  - Se reportan MAE, MedAE y RMSE sobre el test.

Nota: los modelos de boosting entrenan con pérdida MAE (L1); los demás con su pérdida
nativa, pero TODOS se seleccionan y se rankean por MAE, así la comparación es pareja.

Correr: uv run benchmark.py
"""
import warnings, time, json
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from sklearn.model_selection import RandomizedSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, median_absolute_error, mean_squared_error
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

RNG = 42
N_ITER = 50
CUT = pd.Timestamp("2025-09-26")  # test = último año

# ---------- datos ----------
L = pd.read_parquet("landmarks_full.parquet")
raw = pd.read_csv("../dataset/dataset_albahaca_1.csv", parse_dates=["FECHA"])
starts = raw.groupby("ID_COSECHA")["FECHA"].min().rename("start")
L = L.merge(starts, on="ID_COSECHA", how="left")

FEATS = [c for c in L.columns if c not in ["ID_COSECHA", "ID_TANDA", "D", "start"]]
tr = L[L["start"] < CUT]
te = L[L["start"] >= CUT]
Xtr, ytr, gtr = tr[FEATS], tr["D"].to_numpy(float), tr["ID_COSECHA"].to_numpy()
Xte, yte = te[FEATS], te["D"].to_numpy(float)

print(f"Features: {len(FEATS)}")
print(f"TRAIN: {len(tr)} landmarks | {tr['ID_COSECHA'].nunique()} cosechas")
print(f"TEST : {len(te)} landmarks | {te['ID_COSECHA'].nunique()} cosechas "
      f"({te['start'].min().date()} -> {te['start'].max().date()})")

# baseline dummy: duración media del train (constante)
dummy_pred = np.full(len(yte), ytr.mean())
print(f"\nDummy (D=media train): MAE {mean_absolute_error(yte, dummy_pred):.2f} | "
      f"RMSE {np.sqrt(mean_squared_error(yte, dummy_pred)):.2f}")

# ---------- modelos ----------
def pipe(model, scale=False, impute=True):
    steps = []
    if impute:
        steps.append(("imp", SimpleImputer(strategy="median")))
    if scale:
        steps.append(("sc", StandardScaler()))
    steps.append(("m", model))
    return Pipeline(steps)

MODELOS = {
    "ElasticNet (lineal)": (
        pipe(ElasticNet(max_iter=5000, random_state=RNG), scale=True),
        {"m__alpha": [0.001, 0.01, 0.1, 1, 10],
         "m__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]}),
    "kNN (instancias)": (
        pipe(KNeighborsRegressor(), scale=True),
        {"m__n_neighbors": [5, 10, 20, 35, 50, 75],
         "m__weights": ["uniform", "distance"],
         "m__p": [1, 2]}),
    "RandomForest (bagging)": (
        pipe(RandomForestRegressor(n_jobs=1, random_state=RNG), impute=True),
        {"m__n_estimators": [300, 500, 800],
         "m__max_depth": [None, 10, 20, 30],
         "m__min_samples_leaf": [1, 5, 10, 20],
         "m__max_features": [0.3, 0.5, 0.7, 1.0]}),
    "LightGBM (boosting)": (
        pipe(LGBMRegressor(objective="regression_l1", subsample_freq=1, n_jobs=1,
                           verbose=-1, random_state=RNG), impute=False),
        {"m__n_estimators": [300, 500, 800],
         "m__learning_rate": [0.02, 0.03, 0.05, 0.08],
         "m__num_leaves": [15, 31, 63, 127],
         "m__max_depth": [-1, 4, 6, 8],
         "m__min_child_samples": [20, 40, 60, 100],
         "m__subsample": [0.7, 0.8, 1.0],
         "m__colsample_bytree": [0.6, 0.8, 1.0],
         "m__reg_lambda": [0, 1, 3, 5]}),
    "XGBoost (boosting)": (
        pipe(XGBRegressor(objective="reg:absoluteerror", tree_method="hist",
                          n_jobs=1, verbosity=0, random_state=RNG), impute=False),
        {"m__n_estimators": [300, 500, 800],
         "m__learning_rate": [0.02, 0.03, 0.05, 0.08],
         "m__max_depth": [3, 4, 5, 6, 8],
         "m__min_child_weight": [1, 5, 10, 20],
         "m__subsample": [0.7, 0.8, 1.0],
         "m__colsample_bytree": [0.6, 0.8, 1.0],
         "m__reg_lambda": [0.5, 1, 3, 5],
         "m__reg_alpha": [0, 0.5, 1]}),
    "CatBoost (boosting)": (
        pipe(CatBoostRegressor(loss_function="MAE", verbose=0, thread_count=1,
                               random_state=RNG, allow_writing_files=False), impute=False),
        {"m__depth": [4, 6, 8, 10],
         "m__learning_rate": [0.02, 0.03, 0.05, 0.08],
         "m__iterations": [300, 500, 800],
         "m__l2_leaf_reg": [1, 3, 5, 7],
         "m__subsample": [0.7, 0.8, 1.0]}),
    "MLP (red neuronal)": (
        pipe(MLPRegressor(max_iter=300, early_stopping=True, random_state=RNG), scale=True),
        {"m__hidden_layer_sizes": [(64,), (128,), (64, 32), (128, 64)],
         "m__alpha": [1e-4, 1e-3, 1e-2],
         "m__learning_rate_init": [1e-3, 5e-3],
         "m__activation": ["relu", "tanh"]}),
}

inner = GroupKFold(n_splits=5)
resultados = []
for nombre, (est, space) in MODELOS.items():
    grid_size = int(np.prod([len(v) for v in space.values()]))
    n_iter = min(N_ITER, grid_size)
    t0 = time.time()
    print(f"\n>>> {nombre}: {n_iter} configs x 5 folds (grid={grid_size})...", flush=True)
    search = RandomizedSearchCV(est, space, n_iter=n_iter, cv=inner,
                                scoring="neg_mean_absolute_error", n_jobs=-1,
                                random_state=RNG, refit=True)
    search.fit(Xtr, ytr, groups=gtr)
    pred = search.best_estimator_.predict(Xte)
    mae = mean_absolute_error(yte, pred)
    medae = median_absolute_error(yte, pred)
    rmse = np.sqrt(mean_squared_error(yte, pred))
    cv_mae = -search.best_score_
    dt = time.time() - t0
    resultados.append({"modelo": nombre, "test_MAE": mae, "test_MedAE": medae,
                       "test_RMSE": rmse, "cv_MAE_interno": cv_mae, "seg": dt})
    print(f"    TEST -> MAE {mae:.3f} | MedAE {medae:.3f} | RMSE {rmse:.3f}  "
          f"(CV interno MAE {cv_mae:.3f}) [{dt:.0f}s]")
    print(f"    best: {search.best_params_}", flush=True)

# ---------- ranking ----------
res = pd.DataFrame(resultados).sort_values("test_MAE").reset_index(drop=True)
print("\n" + "=" * 78)
print("RANKING — TEST TEMPORAL (último año, 42 cosechas). Ordenado por MAE.")
print("=" * 78)
print(f"{'#':>2} {'Modelo':26s} {'MAE':>7} {'MedAE':>7} {'RMSE':>7} {'CV-int':>7} {'seg':>6}")
print(f"{'':>2} {'Dummy (media)':26s} {mean_absolute_error(yte, dummy_pred):7.2f} "
      f"{median_absolute_error(yte, dummy_pred):7.2f} "
      f"{np.sqrt(mean_squared_error(yte, dummy_pred)):7.2f} {'—':>7} {'—':>6}")
for i, r in res.iterrows():
    print(f"{i+1:>2} {r['modelo']:26s} {r['test_MAE']:7.2f} {r['test_MedAE']:7.2f} "
          f"{r['test_RMSE']:7.2f} {r['cv_MAE_interno']:7.2f} {r['seg']:6.0f}")

res.to_csv("benchmark_resultados.csv", index=False)
print("\nGuardado: benchmark_resultados.csv")
print(f"CAMPEÓN: {res.iloc[0]['modelo']} (test MAE {res.iloc[0]['test_MAE']:.3f} d)")
