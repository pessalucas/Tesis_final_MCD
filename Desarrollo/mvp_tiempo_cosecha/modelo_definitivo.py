"""Modelo definitivo + ablaciones por grupo de features.

Combina todo: trayectoria (base) + clima + historia de duraciones, con LightGBM tuneado.
Mide el aporte de cada grupo por ablación, OOF GroupKFold por cosecha. Guarda el dataset
completo (con historia) y las predicciones OOF del campeón para downstream (conformal, SHAP).
"""
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
from lightgbm import LGBMRegressor

RNG = 42
L = pd.read_parquet("landmarks_clima.parquet")
raw = pd.read_csv("../dataset/dataset_albahaca_1.csv", parse_dates=["FECHA"])

# --- features causales de historia (a nivel tanda) ---
starts = raw.groupby(["ID_COSECHA","ID_TANDA"])["FECHA"].min()
tm = L.drop_duplicates(["ID_COSECHA","ID_TANDA"]).set_index(["ID_COSECHA","ID_TANDA"])[["D","mes_inicio"]].copy()
tm["start"] = starts.reindex(tm.index); tm["end"] = tm["start"] + pd.to_timedelta(tm["D"], unit="D")
tm = tm.sort_values("start"); glob = tm["D"].mean()
H=[]
for idx,row in tm.iterrows():
    ya = tm[tm["end"]<row["start"]]; mm = ya[ya["mes_inicio"]==row["mes_inicio"]]; rec = ya.sort_values("end").tail(5)
    H.append({"ID_COSECHA":idx[0],"ID_TANDA":idx[1],
        "hist_dur_mismomes": mm["D"].mean() if len(mm) else (ya["D"].mean() if len(ya) else glob),
        "hist_n_mismomes": len(mm), "hist_dur_reciente5": rec["D"].mean() if len(rec) else glob,
        "hist_dur_global": ya["D"].mean() if len(ya) else glob,
        "hist_dur_sd_mismomes": mm["D"].std() if len(mm)>1 else 0.0})
L = L.merge(pd.DataFrame(H), on=["ID_COSECHA","ID_TANDA"], how="left")
L.to_parquet("landmarks_full.parquet")

STATIC = ["mes_inicio","mes_sin","mes_cos","cultivo_cod","fraccion_cod","dia_actual"]
CLIMA = [c for c in L.columns if c.startswith("cl_")]
HIST = [c for c in L.columns if c.startswith("hist_")]
TRAJ = [c for c in L.columns if c not in STATIC+CLIMA+HIST+["ID_COSECHA","ID_TANDA","D"]]
BASE = STATIC+TRAJ
y = L["D"].to_numpy(float); groups = L["ID_COSECHA"].to_numpy()
LGB = dict(objective="regression_l1", n_estimators=800, learning_rate=0.03, num_leaves=63,
           max_depth=-1, min_child_samples=20, subsample=0.7, subsample_freq=1,
           colsample_bytree=0.6, reg_lambda=5, random_state=RNG, n_jobs=-1, verbose=-1)

SETS = {"base (trayectoria)": BASE, "base+clima": BASE+CLIMA, "base+historia": BASE+HIST,
        "base+clima+historia (DEFINITIVO)": BASE+CLIMA+HIST}
oof = {k: np.zeros(len(L)) for k in SETS}; fold = {k: [] for k in SETS}
for tr,te in GroupKFold(5).split(L,y,groups):
    for k,f in SETS.items():
        oof[k][te] = LGBMRegressor(**LGB).fit(L.iloc[tr][f], y[tr]).predict(L.iloc[te][f])
        fold[k].append(mean_absolute_error(y[te], oof[k][te]))

print("=== MODELO DEFINITIVO — ablación por grupo (LightGBM tuneado, MAE OOF) ===")
for k in SETS:
    fm=np.array(fold[k])
    print(f"  {k:34s} MAE={mean_absolute_error(y,oof[k]):.3f} | MedAE={np.median(np.abs(y-oof[k])):.2f} | folds {fm.mean():.2f}±{fm.std():.2f}")
np.save("oof_definitivo.npy", oof["base+clima+historia (DEFINITIVO)"])

# MAE por etapa del campeón
champ = oof["base+clima+historia (DEFINITIVO)"]
print("\nMAE por etapa (campeón):")
for lo,hi in [(3,10),(11,20),(21,30),(31,45),(46,100)]:
    m=((L['dia_actual']>=lo)&(L['dia_actual']<=hi)).values
    if m.sum(): print(f"  día {lo}-{hi}: MAE={mean_absolute_error(y[m],champ[m]):.2f}  (n={int(m.sum())})")

# feature importance (gain) del campeón, entrenado en todo
mfull = LGBMRegressor(**LGB).fit(L[BASE+CLIMA+HIST], y)
imp = pd.Series(mfull.booster_.feature_importance(importance_type="gain"), index=BASE+CLIMA+HIST).sort_values(ascending=False)
print("\nTop 15 features (gain) del campeón:")
for n,v in imp.head(15).items():
    grp = "HIST" if n in HIST else ("CLIMA" if n in CLIMA else "base")
    print(f"  [{grp:5s}] {n:24s} {v:10.0f}")
