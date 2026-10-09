"""Objetivo 2 (reencuadrado): proyectar el tiempo de cosecha en el MOMENTO 0.

Pregunta: ¿se justifica un modelo ESTÁTICO dedicado a T=0, o el campeón CatBoost
consultado al inicio ya alcanza?

Diseño:
  - Punto de predicción "T=0" = primer landmark de cada tanda (día ~3, el primer momento
    en que se predice). Una fila por tanda.
  - Target = duración total D.
  - Features T=0 (solo lo conocido al inicio): estáticas (época, cultivo, fracción) +
    historia de cosechas previas + clima de arranque (prev60 + climatología futura +
    pronóstico). SIN trayectoria de sensores ni agregados del ciclo (no observados aún).
  - Challenge de 7 tecnologías (como benchmark.py) + baselines (dummy, promedio por mes).
  - Incumbente "el actual": CatBoost campeón (features COMPLETAS, entrenado en todos los
    días) evaluado en esos mismos puntos T=0.
  - Test temporal (último año). Métricas MAE / MedAE / RMSE.

Correr: uv run estatico_t0.py
"""
import warnings, time
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
from sklearn.model_selection import RandomizedSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet, Ridge, LinearRegression, GammaRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, median_absolute_error, mean_squared_error
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

RNG = 42; N_ITER = 50; CUT = pd.Timestamp("2025-09-26")
L = pd.read_parquet("landmarks_full.parquet")
raw = pd.read_csv("../dataset/dataset_albahaca_1.csv", parse_dates=["FECHA"])
starts = raw.groupby("ID_COSECHA")["FECHA"].min().rename("start")
L = L.merge(starts, on="ID_COSECHA", how="left")
ALL = [c for c in L.columns if c not in ["ID_COSECHA", "ID_TANDA", "D", "start"]]

# --- features disponibles en T=0 ---
STATIC0 = ["mes_inicio", "mes_sin", "mes_cos", "cultivo_cod", "fraccion_cod"]
HIST = [c for c in ALL if c.startswith("hist_")]
CL_ALL = [c for c in ALL if c.startswith("cl_")]
CL_CYCLE = [c for c in CL_ALL if any(k in c for k in ["_ciclo", "_last", "acum", "ndias", "tendencia", "anom_last"])]
CL_T0 = [c for c in CL_ALL if c not in CL_CYCLE]  # prev60 + fut*_clim + fc_* (conocidos al inicio)
FEATS_T0 = STATIC0 + HIST + CL_T0
print(f"Features T=0: {len(FEATS_T0)} (estáticas {len(STATIC0)} + historia {len(HIST)} + clima-inicio {len(CL_T0)})")
print(f"  clima-inicio: {CL_T0}")

# --- snapshot T=0: primer landmark por tanda ---
idx = L.groupby(["ID_COSECHA", "ID_TANDA"])["dia_actual"].idxmin()
T0 = L.loc[idx].copy()
print(f"Filas T=0 (una por tanda): {len(T0)} | día actual medio en T=0: {T0['dia_actual'].mean():.1f}")
tr0, te0 = T0[T0["start"] < CUT], T0[T0["start"] >= CUT]
ytr, gtr = tr0["D"].to_numpy(float), tr0["ID_COSECHA"].to_numpy()
yte = te0["D"].to_numpy(float)
print(f"TRAIN T=0: {len(tr0)} tandas | TEST T=0: {len(te0)} tandas ({te0['ID_COSECHA'].nunique()} cosechas)\n")

def M(yt, p): return mean_absolute_error(yt, p), median_absolute_error(yt, p), np.sqrt(mean_squared_error(yt, p))
res = []

# --- baselines ---
res.append(("Dummy (media D)", *M(yte, np.full(len(yte), ytr.mean()))))
mes_mean = tr0.groupby("mes_inicio")["D"].mean()
res.append(("Promedio por mes", *M(yte, te0["mes_inicio"].map(mes_mean).fillna(ytr.mean()).to_numpy())))

# --- challenge de tecnologías (features T=0) ---
def pipe(m, scale=False, impute=True):
    st = []
    if impute: st.append(("imp", SimpleImputer(strategy="median")))
    if scale: st.append(("sc", StandardScaler()))
    st.append(("m", m)); return Pipeline(st)

MODELOS = {
    # --- modelos SIMPLES / baja complejidad (apropiados para muestra chica) ---
    "LinearReg (OLS)": (pipe(LinearRegression(), scale=True), {}),
    "Ridge": (pipe(Ridge(random_state=RNG), scale=True),
        {"m__alpha": [1, 10, 50, 100, 300, 1000]}),
    "GLM Gamma": (pipe(GammaRegressor(max_iter=10000), scale=True),
        {"m__alpha": [0.001, 0.01, 0.1, 1, 10]}),
    "Árbol simple (shallow)": (pipe(DecisionTreeRegressor(random_state=RNG)),
        {"m__max_depth": [2, 3, 4, 5], "m__min_samples_leaf": [10, 20, 40]}),
    # --- modelos más complejos ---
    "ElasticNet": (pipe(ElasticNet(max_iter=5000, random_state=RNG), scale=True),
        {"m__alpha": [0.001, 0.01, 0.1, 1, 10], "m__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]}),
    "kNN": (pipe(KNeighborsRegressor(), scale=True),
        {"m__n_neighbors": [5, 10, 20, 35, 50], "m__weights": ["uniform", "distance"], "m__p": [1, 2]}),
    "RandomForest": (pipe(RandomForestRegressor(n_jobs=1, random_state=RNG)),
        {"m__n_estimators": [300, 500, 800], "m__max_depth": [None, 6, 10, 20],
         "m__min_samples_leaf": [1, 3, 5, 10], "m__max_features": [0.3, 0.5, 0.7, 1.0]}),
    "LightGBM": (pipe(LGBMRegressor(objective="regression_l1", n_jobs=1, verbose=-1, random_state=RNG), impute=False),
        {"m__n_estimators": [200, 400, 800], "m__learning_rate": [0.02, 0.03, 0.05, 0.08],
         "m__num_leaves": [7, 15, 31, 63], "m__min_child_samples": [10, 20, 40, 60],
         "m__subsample": [0.7, 0.8, 1.0], "m__colsample_bytree": [0.6, 0.8, 1.0], "m__reg_lambda": [0, 1, 3, 5]}),
    "XGBoost": (pipe(XGBRegressor(objective="reg:absoluteerror", tree_method="hist", n_jobs=1, verbosity=0, random_state=RNG), impute=False),
        {"m__n_estimators": [200, 400, 800], "m__learning_rate": [0.02, 0.03, 0.05, 0.08],
         "m__max_depth": [3, 4, 5, 6], "m__min_child_weight": [1, 5, 10], "m__subsample": [0.7, 0.8, 1.0],
         "m__colsample_bytree": [0.6, 0.8, 1.0], "m__reg_lambda": [0.5, 1, 3, 5]}),
    "CatBoost": (pipe(CatBoostRegressor(loss_function="MAE", verbose=0, thread_count=1, random_state=RNG, allow_writing_files=False), impute=False),
        {"m__depth": [3, 4, 6, 8], "m__learning_rate": [0.02, 0.03, 0.05, 0.08],
         "m__iterations": [200, 400, 800], "m__l2_leaf_reg": [1, 3, 5, 7]}),
    "MLP": (pipe(MLPRegressor(max_iter=500, early_stopping=True, random_state=RNG), scale=True),
        {"m__hidden_layer_sizes": [(32,), (64,), (64, 32)], "m__alpha": [1e-4, 1e-3, 1e-2],
         "m__learning_rate_init": [1e-3, 5e-3], "m__activation": ["relu", "tanh"]}),
}
inner = GroupKFold(n_splits=5)
for name, (est, space) in MODELOS.items():
    t0 = time.time()
    if space:
        n_iter = min(N_ITER, int(np.prod([len(v) for v in space.values()])))
        s = RandomizedSearchCV(est, space, n_iter=n_iter, cv=inner, scoring="neg_mean_absolute_error",
                               n_jobs=-1, random_state=RNG, refit=True)
        s.fit(tr0[FEATS_T0], ytr, groups=gtr)
        model = s.best_estimator_
    else:  # sin hiperparámetros que buscar (ej. OLS) -> ajustar directo
        model = est.fit(tr0[FEATS_T0], ytr)
    res.append((f"{name} [T=0]", *M(yte, model.predict(te0[FEATS_T0]))))
    print(f"  {name:22s} done [{time.time()-t0:.0f}s]", flush=True)

# --- incumbente: campeón CatBoost (features COMPLETAS, todos los días) evaluado en T=0 ---
BEST = dict(loss_function="MAE", depth=6, learning_rate=0.03, l2_leaf_reg=1, iterations=500,
            subsample=0.7, random_state=RNG, verbose=0, thread_count=-1, allow_writing_files=False)
Ltr = L[L["start"] < CUT]
champ = CatBoostRegressor(**BEST).fit(Ltr[ALL], Ltr["D"].to_numpy(float))
res.append(("CatBoost CAMPEÓN @T=0 (features completas)", *M(yte, champ.predict(te0[ALL]))))

# --- ranking ---
dfres = pd.DataFrame(res, columns=["modelo", "MAE", "MedAE", "RMSE"]).sort_values("MAE").reset_index(drop=True)
print("\n" + "=" * 74)
print("RANKING T=0 — proyectar duración al inicio del ciclo (test temporal)")
print("=" * 74)
print(f"{'#':>2} {'Modelo':42s} {'MAE':>6} {'MedAE':>6} {'RMSE':>6}")
for i, r in dfres.iterrows():
    print(f"{i+1:>2} {r['modelo']:42s} {r['MAE']:6.2f} {r['MedAE']:6.2f} {r['RMSE']:6.2f}")
dfres.to_csv("estatico_t0_resultados.csv", index=False)
best_t0 = dfres[dfres["modelo"].str.contains(r"\[T=0\]")].iloc[0]
champ_row = dfres[dfres["modelo"].str.contains("CAMPEÓN")].iloc[0]
print(f"\nMejor modelo dedicado T=0: {best_t0['modelo']} (MAE {best_t0['MAE']:.2f})")
print(f"Campeón general @T=0:      MAE {champ_row['MAE']:.2f}")
d = champ_row['MAE'] - best_t0['MAE']
print(f"=> El modelo dedicado {'GANA' if d>0 else 'NO gana'} por {abs(d):.2f} d "
      f"({'se justifica un modelo aparte' if d>0.2 else 'NO se justifica: el campeón ya alcanza'})")
