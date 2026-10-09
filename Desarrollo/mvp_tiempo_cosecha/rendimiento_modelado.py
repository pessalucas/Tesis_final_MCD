"""Objetivo 3 — predecir la cantidad de plantines por cosecha: comparación de enfoques.

Enfoques comparados (target = rendimiento en plantas, cosechas AISLADAS 2025-26, n≈66):
  BASELINES:  media global · promedio por mes (estacional).
  MATEMÁTICO: rendimiento ~ duración (lineal y potencia), usando la duración REAL y la
              duración ESTIMADA por CatBoost (OOF) -> ¿alcanza una fórmula sobre CatBoost?
  MODELADO DIRECTO: challenge de modelos simples (dado el n chico) sobre features T=0.
  COMBINADO: features T=0 + duración estimada por CatBoost.
  UPPER BOUND: features del ciclo completo (retrospectivo, referencia de techo).
  FORECASTING: volumen mensual como serie temporal (naive estacional + Holt-Winters).

Validación: KFold(5) repetido (MAE/RMSE) + split temporal (train viejo -> test nuevo).
NOTA: target tentativo (atribución por cierre más cercano) y n chico -> exploratorio.

Correr: uv run --with openpyxl python rendimiento_modelado.py
"""
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, GammaRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, median_absolute_error, mean_squared_error
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

RNG = 42

# ---------- features a nivel cosecha ----------
r = pd.read_csv("rendimiento_por_cosecha.csv")
if "cosecha" in r.columns and "ID_COSECHA" not in r.columns:
    r = r.rename(columns={"cosecha": "ID_COSECHA"})
r["inicio"] = pd.to_datetime(r["inicio"]); r["cierre"] = pd.to_datetime(r["cierre"])
r["mes_ini"] = r["inicio"].dt.month
r["mes_sin"] = np.sin(2*np.pi*r["mes_ini"]/12); r["mes_cos"] = np.cos(2*np.pi*r["mes_ini"]/12)

# duración estimada por CatBoost (OOF) + clima por cosecha, desde landmarks_full
L = pd.read_parquet("landmarks_full.parquet")
oof = np.load("oof_catboost.npy"); L = L.assign(oof=oof)
first = L.loc[L.groupby(["ID_COSECHA","ID_TANDA"])["dia_actual"].idxmin()]
cb_t0 = first.groupby("ID_COSECHA")["oof"].mean().rename("dur_cb_t0").reset_index()
clima_ciclo = L.groupby("ID_COSECHA").agg(cl_dli_ciclo=("cl_dli_mean_ciclo","mean")).reset_index()
clima_t0 = first.groupby("ID_COSECHA").agg(cl_prev60_dli=("cl_prev60_dli","mean"),
                                           cl_fut30_dli=("cl_fut30_dli_clim","mean")).reset_index()
r = r.merge(cb_t0, on="ID_COSECHA", how="left").merge(clima_ciclo, on="ID_COSECHA", how="left") \
     .merge(clima_t0, on="ID_COSECHA", how="left")

# set de modelado: cosechas aisladas con rendimiento>0 (etiqueta más limpia)
m = r[(r["aislada"]) & (r["rendimiento"] > 0)].copy().sort_values("cierre").reset_index(drop=True)
y = m["rendimiento"].to_numpy(float)
print(f"Cosechas para modelar (aisladas, rend>0): {len(m)} | rendimiento medio {y.mean():.0f} plantas")
print(f"Correlaciones (Spearman) con el rendimiento:")
for c in ["dur", "dur_cb_t0", "mes_sin", "mes_cos", "n_tandas", "cl_dli_ciclo", "cl_fut30_dli", "cl_prev60_dli"]:
    print(f"  {c:14s} {m[[c,'rendimiento']].corr(method='spearman').iloc[0,1]:+.3f}")

FSETS = {
    "SEASON":        ["mes_sin", "mes_cos"],
    "DUR_real":      ["dur"],
    "DUR_catboost":  ["dur_cb_t0"],
    "T0":            ["mes_sin", "mes_cos", "n_tandas", "cl_fut30_dli", "cl_prev60_dli"],
    "T0+DUR_cb":     ["mes_sin", "mes_cos", "n_tandas", "cl_fut30_dli", "cl_prev60_dli", "dur_cb_t0"],
    "FULL_ciclo":    ["mes_sin", "mes_cos", "n_tandas", "cl_fut30_dli", "cl_prev60_dli", "dur", "cl_dli_ciclo"],
}

def pipe(mdl, scale=False):
    st = [("imp", SimpleImputer(strategy="median"))]
    if scale: st.append(("sc", StandardScaler()))
    st.append(("m", mdl)); return Pipeline(st)

MODELS = {
    "OLS":        pipe(LinearRegression(), scale=True),
    "Ridge":      pipe(Ridge(alpha=10, random_state=RNG), scale=True),
    "GLM Gamma":  pipe(GammaRegressor(alpha=0.1, max_iter=10000), scale=True),
    "Árbol(sh)":  pipe(DecisionTreeRegressor(max_depth=3, min_samples_leaf=8, random_state=RNG)),
    "kNN":        pipe(KNeighborsRegressor(n_neighbors=10, weights="distance"), scale=True),
    "RandomForest": pipe(RandomForestRegressor(n_estimators=400, min_samples_leaf=5, random_state=RNG, n_jobs=1)),
    "LightGBM":   pipe(LGBMRegressor(objective="regression_l1", n_estimators=300, learning_rate=0.03,
                                     num_leaves=15, min_child_samples=10, reg_lambda=3, n_jobs=1, verbose=-1, random_state=RNG)),
    "CatBoost":   pipe(CatBoostRegressor(loss_function="MAE", depth=3, learning_rate=0.03, iterations=400,
                                         l2_leaf_reg=5, verbose=0, thread_count=1, random_state=RNG, allow_writing_files=False)),
}

cv = KFold(n_splits=5, shuffle=True, random_state=RNG)
def M(yt, p): return mean_absolute_error(yt, p), median_absolute_error(yt, p), np.sqrt(mean_squared_error(yt, p))
res = []

# baselines
res.append(("Baseline: media global", "-", *M(y, np.full(len(y), y.mean()))))
mes_mean = m.groupby("mes_ini")["rendimiento"].transform("mean")
res.append(("Baseline: promedio por mes", "-", *M(y, mes_mean.to_numpy())))

# matemático: rendimiento ~ duración (potencia) con dur real y dur CatBoost
for dcol, lab in [("dur", "MAT: potencia ~ dur_real"), ("dur_cb_t0", "MAT: potencia ~ dur_CatBoost")]:
    x = np.log(m[dcol].clip(lower=1).to_numpy()); ly = np.log(y)
    b, a = np.polyfit(x, ly, 1)  # log-log => potencia
    pred = np.exp(a + b * x)
    res.append((lab, dcol, *M(y, pred)))

# challenge de modelos x feature set (KFold CV)
for fs, cols in FSETS.items():
    X = m[cols]
    for name, mdl in MODELS.items():
        try:
            pred = cross_val_predict(mdl, X, y, cv=cv, n_jobs=-1)
            res.append((f"ML: {name}", fs, *M(y, pred)))
        except Exception as e:
            res.append((f"ML: {name}", fs, np.nan, np.nan, np.nan))

df = pd.DataFrame(res, columns=["enfoque", "features", "MAE", "MedAE", "RMSE"]).sort_values("MAE").reset_index(drop=True)
pd.set_option("display.width", 120)
print("\n" + "=" * 78)
print(f"RENDIMIENTO — ranking de enfoques (MAE en plantas; media real {y.mean():.0f})")
print("=" * 78)
print(df.head(25).to_string(index=False, float_format=lambda x: f"{x:.0f}"))
df.to_csv("rendimiento_modelado_resultados.csv", index=False)
best = df.iloc[0]
print(f"\nMejor: {best['enfoque']} [{best['features']}] -> MAE {best['MAE']:.0f} plantas "
      f"({best['MAE']/y.mean()*100:.0f}% del promedio)")

# ---------- FORECASTING: volumen mensual ----------
print("\n" + "=" * 78)
print("FORECASTING de volumen mensual (serie temporal)")
print("=" * 78)
import openpyxl
wb = openpyxl.load_workbook("../dataset/HGP - Albahacas por cosecha.xlsx", read_only=True, data_only=True)
ws = wb["Hoja 1"]; rows = list(ws.iter_rows(values_only=True))[1:]
recs = []
for rr in rows:
    if rr[0] is not None: recs.append((rr[0], rr[1]))
    if len(rr) > 4 and rr[3] is not None: recs.append((rr[3], rr[4]))
dd = pd.DataFrame(recs, columns=["fecha", "cant"]); dd["fecha"] = pd.to_datetime(dd["fecha"])
dd["cant"] = pd.to_numeric(dd["cant"], errors="coerce"); dd = dd.dropna()
mens = dd.set_index("fecha")["cant"].resample("ME").sum()
mens = mens[mens > 0]
print(f"Serie mensual: {len(mens)} meses ({mens.index.min().date()} -> {mens.index.max().date()})")
# naive estacional: mismo mes del año anterior; test = últimos 8 meses con y-1 disponible
test = mens.iloc[-8:]
naive = pd.Series([mens.get(t - pd.DateOffset(years=1), np.nan) for t in test.index], index=test.index)
mask = naive.notna()
if mask.sum():
    print(f"Naive estacional (mismo mes año previo), {int(mask.sum())} meses test: "
          f"MAE {mean_absolute_error(test[mask], naive[mask]):.0f} | "
          f"MAPE {np.mean(np.abs((test[mask]-naive[mask])/test[mask]))*100:.0f}%")
try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    tr = mens.iloc[:-8]
    hw = ExponentialSmoothing(tr, trend="add", seasonal="add", seasonal_periods=12).fit()
    fc = hw.forecast(8)
    print(f"Holt-Winters (aditivo, 8 meses test): MAE {mean_absolute_error(test, fc.reindex(test.index)):.0f}")
except Exception as e:
    print(f"Holt-Winters: no disponible ({type(e).__name__}); requiere statsmodels + >=2 ciclos.")
print("\nLISTO.")
