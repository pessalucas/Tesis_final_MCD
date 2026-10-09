"""¿Por qué cl_fc_dli_next7 es top feature pero el oráculo no da lift?

Hipótesis: la feature es importante porque codifica el NIVEL ESTACIONAL de luz (redundante
con calendario/climatología), no por skill de pronóstico. Se mide con:
  1. correlación de cl_fc_dli_next7 con proxies de estación.
  2. drop-column: MAE quitando cl_fc_dli_next7 / todas las de FUTURO / todo el clima.
Si al quitarla el MAE casi no cambia -> era redundante (importancia != lift).
"""
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
from lightgbm import LGBMRegressor

RNG=42
L=pd.read_parquet("landmarks_full.parquet")
CLIMA=[c for c in L.columns if c.startswith("cl_")]
FUT=[c for c in CLIMA if ("fc_" in c) or ("fut" in c) or ("tendencia" in c)]  # features de futuro
ALL=[c for c in L.columns if c not in ["ID_COSECHA","ID_TANDA","D"]]
y=L["D"].to_numpy(float);groups=L["ID_COSECHA"].to_numpy()

# 1) correlaciones
print("=== ¿cl_fc_dli_next7 es un proxy de estación? ===")
for c in ["cl_fut7_dli_clim","cl_dli_mean_ciclo","mes_cos","cl_prev60_dli"]:
    print(f"  corr(cl_fc_dli_next7, {c:20s}) = {L['cl_fc_dli_next7'].corr(L[c]):+.2f}")

# 2) drop-column
LGB=dict(objective="regression_l1",n_estimators=800,learning_rate=0.03,num_leaves=63,max_depth=-1,
    min_child_samples=20,subsample=0.7,subsample_freq=1,colsample_bytree=0.6,reg_lambda=5,random_state=RNG,n_jobs=-1,verbose=-1)
def cv(feats):
    oof=np.zeros(len(L))
    for tr,te in GroupKFold(5).split(L,y,groups):
        oof[te]=LGBMRegressor(**LGB).fit(L.iloc[tr][feats],y[tr]).predict(L.iloc[te][feats])
    return mean_absolute_error(y,oof)

base=cv(ALL)
print("\n=== Drop-column (MAE OOF; aporte MARGINAL = cuánto sube al quitar) ===")
print(f"  campeón (todo)                          MAE={base:.3f}")
for etq,drop in [("sin cl_fc_dli_next7", ["cl_fc_dli_next7"]),
                 ("sin TODAS las de futuro (fc+fut+tend)", FUT),
                 ("sin NADA de clima", CLIMA)]:
    feats=[c for c in ALL if c not in drop]
    m=cv(feats)
    print(f"  {etq:38s} MAE={m:.3f}  (Δ={m-base:+.3f})")
