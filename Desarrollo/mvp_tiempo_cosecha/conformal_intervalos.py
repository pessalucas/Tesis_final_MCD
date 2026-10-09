"""Conformal prediction (CQR) para intervalos con cobertura GARANTIZADA.

Los intervalos por cuantiles quedaron sub-cubiertos (56% vs 80%). CQR (Conformalized
Quantile Regression) corrige eso: usa un set de calibración para ajustar el ancho y
garantizar la cobertura nominal, manteniendo la adaptabilidad (más angosto al avanzar).

Split-conformal agrupado por cosecha: en cada fold de test, del resto se separan
cosechas de TRAIN (ajustan los cuantiles) y de CALIBRACIÓN (fijan el ajuste E).
"""
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
from sklearn.model_selection import GroupKFold
from lightgbm import LGBMRegressor

RNG = 42
L = pd.read_parquet("landmarks_full.parquet")
L["restante"] = L["D"] - L["dia_actual"]
FEATS = [c for c in L.columns if c not in ["ID_COSECHA","ID_TANDA","D","restante"]]
y = L["restante"].to_numpy(float); groups = L["ID_COSECHA"].to_numpy()
ALPHA = 0.10  # intervalo (1-2*alpha)=80%
def q(a): return LGBMRegressor(objective="quantile", alpha=a, n_estimators=400, learning_rate=0.05,
    num_leaves=63, min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
    reg_lambda=3, random_state=RNG, n_jobs=-1, verbose=-1)

rng = np.random.default_rng(RNG)
lo_naive=np.zeros(len(L)); hi_naive=np.zeros(len(L)); lo_cqr=np.zeros(len(L)); hi_cqr=np.zeros(len(L))
for tr_all, te in GroupKFold(5).split(L, y, groups):
    cos = pd.unique(groups[tr_all]); rng.shuffle(cos)
    n_cal = int(len(cos)*0.3); cal_cos = set(cos[:n_cal])
    cal = tr_all[np.isin(groups[tr_all], list(cal_cos))]
    tr  = tr_all[~np.isin(groups[tr_all], list(cal_cos))]
    mlo = q(ALPHA).fit(L.iloc[tr][FEATS], y[tr]); mhi = q(1-ALPHA).fit(L.iloc[tr][FEATS], y[tr])
    # calibración: conformity score de CQR
    lo_c = mlo.predict(L.iloc[cal][FEATS]); hi_c = mhi.predict(L.iloc[cal][FEATS])
    s = np.maximum(lo_c - y[cal], y[cal] - hi_c)
    k = int(np.ceil((len(s)+1)*(1-2*ALPHA))); k = min(k, len(s))
    E = np.sort(s)[k-1]
    # test
    lo_t = mlo.predict(L.iloc[te][FEATS]); hi_t = mhi.predict(L.iloc[te][FEATS])
    lo_naive[te], hi_naive[te] = lo_t, hi_t
    lo_cqr[te], hi_cqr[te] = lo_t - E, hi_t + E

def cov(lo,hi): return np.mean((y>=lo)&(y<=hi))*100
def width(lo,hi): return np.mean(hi-lo)
print("=== Intervalos 80%: NAIVE (cuantiles) vs CQR (conformal) ===")
print(f"  {'':10s} {'cobertura':>10s} {'ancho medio':>12s}")
print(f"  {'naive':10s} {cov(lo_naive,hi_naive):9.1f}% {width(lo_naive,hi_naive):11.1f}d")
print(f"  {'CQR':10s} {cov(lo_cqr,hi_cqr):9.1f}% {width(lo_cqr,hi_cqr):11.1f}d   (objetivo 80%)")
print("\nCobertura y ancho CQR por etapa:")
for lo,hi in [(3,10),(11,20),(21,30),(31,45),(46,100)]:
    m=((L['dia_actual']>=lo)&(L['dia_actual']<=hi)).values
    if m.sum(): print(f"  día {lo}-{hi}: cob={np.mean((y[m]>=lo_cqr[m])&(y[m]<=hi_cqr[m]))*100:.1f}%  ancho={np.mean(hi_cqr[m]-lo_cqr[m]):.1f}d")
