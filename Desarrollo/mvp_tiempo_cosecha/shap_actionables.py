"""SHAP sobre el campeón — foco en FACTORES ACCIONABLES por el operador.

Usa los SHAP nativos de LightGBM (pred_contrib) — sin dependencia extra. Separa las
variables CONTROLABLES por el operador (pH, PPM/nutrientes, eventos de manejo) de las
no controlables (clima, historia, estación), y para las accionables reporta magnitud y
SENTIDO del efecto sobre la duración predicha -> recomendaciones prácticas.
"""
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
from lightgbm import LGBMRegressor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG=42
L=pd.read_parquet("landmarks_full.parquet")
FEATS=[c for c in L.columns if c not in ["ID_COSECHA","ID_TANDA","D"]]
LGB=dict(objective="regression_l1",n_estimators=800,learning_rate=0.03,num_leaves=63,max_depth=-1,
    min_child_samples=20,subsample=0.7,subsample_freq=1,colsample_bytree=0.6,reg_lambda=5,
    random_state=RNG,n_jobs=-1,verbose=-1)
m=LGBMRegressor(**LGB).fit(L[FEATS],L["D"])
sv=m.predict(L[FEATS],pred_contrib=True)[:,:-1]  # última col = base value
S=pd.DataFrame(sv,columns=FEATS)

# clasificar accionables (controlables por el operador)
ACC=[c for c in FEATS if c.startswith(("ph_","ppm_","n_","dias_desde_","tiene_tanque","dias_en_tanque"))]
NOACC=[c for c in FEATS if c not in ACC]
def signo(f):  # correlación SHAP vs valor -> sentido del efecto
    v=L[f].values; s=S[f].values; mask=~np.isnan(v)
    if mask.sum()<10 or np.nanstd(v[mask])==0: return 0.0
    return np.corrcoef(v[mask],s[mask])[0,1]

imp=S.abs().mean().sort_values(ascending=False)
print("=== Importancia SHAP: accionables vs no-accionables ===")
print(f"  Suma |SHAP| accionables:    {S[ACC].abs().mean().sum():.2f}")
print(f"  Suma |SHAP| no-accionables: {S[NOACC].abs().mean().sum():.2f}")
print("\n=== Top FACTORES ACCIONABLES (magnitud y sentido sobre la duración) ===")
print(f"  {'feature':22s} {'|SHAP| medio':>12s} {'sentido':>9s}  interpretación")
for f in [x for x in imp.index if x in ACC][:10]:
    sg=signo(f)
    txt = "↑valor → ciclo MÁS LARGO" if sg>0.05 else ("↑valor → ciclo MÁS CORTO" if sg<-0.05 else "efecto no monótono")
    print(f"  {f:22s} {imp[f]:12.3f} {sg:+9.2f}  {txt}")

# dependence plots de las 6 accionables top
top6=[x for x in imp.index if x in ACC][:6]
fig,ax=plt.subplots(2,3,figsize=(15,8))
for a,f in zip(ax.ravel(),top6):
    a.scatter(L[f],S[f],s=4,alpha=0.3,color="#2f7de8")
    a.axhline(0,color="k",lw=0.7); a.set_xlabel(f); a.set_ylabel("SHAP (días)")
    a.set_title(f"efecto de {f} en la duración",fontsize=9); a.grid(alpha=0.3)
fig.suptitle("Factores accionables: efecto (SHAP) sobre la duración predicha",fontsize=13)
fig.tight_layout(); fig.savefig("shap_accionables.png",dpi=120)
print("\nFigura: shap_accionables.png")
