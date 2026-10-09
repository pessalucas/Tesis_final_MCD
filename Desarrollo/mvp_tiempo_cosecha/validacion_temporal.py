"""Validación TEMPORAL (train pasado -> test futuro) = realismo de producción.

La CV agrupada mezcla cosechas de distintos años. En producción se predicen cosechas
FUTURAS con un modelo entrenado en el pasado. Este test entrena en las cosechas más
tempranas y evalúa en las más tardías (expanding window). Si el MAE se mantiene cerca
del de la CV agrupada (2.80), el modelo generaliza bien hacia adelante.
"""
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
from sklearn.metrics import mean_absolute_error
from lightgbm import LGBMRegressor

RNG=42
L = pd.read_parquet("landmarks_full.parquet")
raw = pd.read_csv("../dataset/dataset_albahaca_1.csv", parse_dates=["FECHA"])
starts = raw.groupby(["ID_COSECHA","ID_TANDA"])["FECHA"].min().rename("start").reset_index()
L = L.merge(starts, on=["ID_COSECHA","ID_TANDA"], how="left")
FEATS=[c for c in L.columns if c not in ["ID_COSECHA","ID_TANDA","D","start"]]
LGB=dict(objective="regression_l1",n_estimators=800,learning_rate=0.03,num_leaves=63,max_depth=-1,
    min_child_samples=20,subsample=0.7,subsample_freq=1,colsample_bytree=0.6,reg_lambda=5,
    random_state=RNG,n_jobs=-1,verbose=-1)

# cosechas ordenadas por fecha de inicio
cos_start = L.groupby("ID_COSECHA")["start"].min().sort_values()
cos_ord = cos_start.index.tolist()
print("=== Validación temporal (expanding window) ===")
print(f"  {'train':>18s} {'test (futuro)':>28s} {'n_test_cos':>10s} {'MAE':>7s}")
for frac in [0.5,0.6,0.7,0.8]:
    ncut=int(len(cos_ord)*frac)
    tr_cos=set(cos_ord[:ncut]); te_cos=set(cos_ord[ncut:])
    trm=L["ID_COSECHA"].isin(tr_cos).values; tem=L["ID_COSECHA"].isin(te_cos).values
    m=LGBMRegressor(**LGB).fit(L[trm][FEATS],L[trm]["D"])
    mae=mean_absolute_error(L[tem]["D"],m.predict(L[tem][FEATS]))
    t_ini=cos_start[cos_ord[:ncut]].max().date(); t_fin=cos_start[cos_ord[ncut:]].max().date()
    print(f"  hasta {str(t_ini):>12s} {'-> '+str(cos_start[cos_ord[ncut]].date())+'..'+str(t_fin):>28s} {len(te_cos):10d} {mae:6.2f}d")
print("\n  Referencia CV agrupada (no temporal): 2.80 d")
