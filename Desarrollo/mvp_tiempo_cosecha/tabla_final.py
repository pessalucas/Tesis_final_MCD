"""Tabla consolidada de todos los modelos y benchmarks.

Todos evaluados igual: GroupKFold(5) por ID_COSECHA, out-of-fold, MAE en días.
Benchmarks (sin ML) calculados solo con el train de cada fold (sin fuga):
  - Dummy            : duración = promedio global de las cosechas del train.
  - Promedio por mes : duración = promedio de las cosechas del train que iniciaron
                       ese mismo mes de inicio.
Modelos: progresión HistGBR (estáticas -> completo -> +clima) + LightGBM tuneado.
"""
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
from lightgbm import LGBMRegressor

RNG = 42
L = pd.read_parquet("landmarks_clima.parquet")
y = L["D"].to_numpy(float)
groups = L["ID_COSECHA"].to_numpy()

STATIC = ["mes_inicio","mes_sin","mes_cos","cultivo_cod","fraccion_cod","dia_actual"]
CLIMA = [c for c in L.columns if c.startswith("cl_")]
TRAJ = [c for c in L.columns if c not in STATIC+CLIMA+["ID_COSECHA","ID_TANDA","D"]]
ALL = STATIC + TRAJ + CLIMA

LGB_PARAMS = dict(objective="regression_l1", n_estimators=800, learning_rate=0.03,
                  num_leaves=63, max_depth=-1, min_child_samples=20, subsample=0.7,
                  subsample_freq=1, colsample_bytree=0.6, reg_lambda=5,
                  random_state=RNG, n_jobs=-1, verbose=-1)
def hgb(): return HistGradientBoostingRegressor(max_depth=4, max_iter=300, learning_rate=0.05,
    min_samples_leaf=40, l2_regularization=1.0, random_state=RNG)

# --- benchmark CAUSAL: promedio del mes de inicio usando SOLO cosechas ya cerradas ---
raw = pd.read_csv("../dataset/dataset_albahaca_1.csv", parse_dates=["FECHA"])
starts = raw.groupby(["ID_COSECHA","ID_TANDA"])["FECHA"].min()
tmeta = L.drop_duplicates(["ID_COSECHA","ID_TANDA"]).set_index(["ID_COSECHA","ID_TANDA"])[["D","mes_inicio"]].copy()
tmeta["start"] = starts.reindex(tmeta.index)
tmeta["end"] = tmeta["start"] + pd.to_timedelta(tmeta["D"], unit="D")
glob = tmeta["D"].mean()
def causal(row):
    ya = tmeta["end"] < row["start"]                       # cerradas antes de que ésta arranque
    m = ya & (tmeta["mes_inicio"] == row["mes_inicio"])    # mismo mes de inicio
    if m.any(): return tmeta.loc[m, "D"].mean()
    return tmeta.loc[ya, "D"].mean() if ya.any() else glob # fallback: prom. de todo lo pasado
tmeta["causal_pred"] = tmeta.apply(causal, axis=1)
causal_pred = L.merge(tmeta["causal_pred"].reset_index(), on=["ID_COSECHA","ID_TANDA"], how="left")["causal_pred"].values

MODELOS = ["Dummy (prom. global)", "Prom. por mes de inicio (causal)",
           "HistGBR estáticas", "HistGBR completo", "HistGBR +clima", "LightGBM tuneado (all)"]
oof = {m: np.zeros(len(L)) for m in MODELOS}
oof["Prom. por mes de inicio (causal)"] = causal_pred
fold = {m: [] for m in MODELOS}

for tr, te in GroupKFold(5).split(L, y, groups):
    tr_tandas = L.iloc[tr].drop_duplicates(["ID_COSECHA","ID_TANDA"])
    # --- dummy (nivel tanda, solo train) ---
    oof["Dummy (prom. global)"][te] = tr_tandas["D"].mean()
    # --- modelos ---
    oof["HistGBR estáticas"][te] = hgb().fit(L.iloc[tr][STATIC], y[tr]).predict(L.iloc[te][STATIC])
    oof["HistGBR completo"][te] = hgb().fit(L.iloc[tr][STATIC+TRAJ], y[tr]).predict(L.iloc[te][STATIC+TRAJ])
    oof["HistGBR +clima"][te] = hgb().fit(L.iloc[tr][ALL], y[tr]).predict(L.iloc[te][ALL])
    oof["LightGBM tuneado (all)"][te] = LGBMRegressor(**LGB_PARAMS).fit(L.iloc[tr][ALL], y[tr]).predict(L.iloc[te][ALL])
    for m in MODELOS:
        fold[m].append(mean_absolute_error(y[te], oof[m][te]))

print("="*72)
print(f"{'MODELO':30s} {'MAE':>6s} {'MedAE':>6s} {'folds (mean±sd)':>18s}  {'vs dummy':>9s}")
print("="*72)
base = mean_absolute_error(y, oof["Dummy (prom. global)"])
for m in MODELOS:
    mae = mean_absolute_error(y, oof[m]); medae = np.median(np.abs(y-oof[m]))
    fm = np.array(fold[m])
    mej = 100*(base-mae)/base
    print(f"{m:30s} {mae:6.2f} {medae:6.2f}   {fm.mean():5.2f} ± {fm.std():4.2f}     {mej:+6.1f}%")
print("="*72)
print(f"Landmarks {len(L)} | cosechas {L['ID_COSECHA'].nunique()} | target = duración total D")
