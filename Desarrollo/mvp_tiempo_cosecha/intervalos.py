"""EXP-A3 · Intervalos de predicción con regresión por cuantiles (LightGBM).

En vez de un número, el modelo dice "faltan X días, entre p10 y p90". Es la forma en
que el dato se usa de verdad para planificar (comprometer una fecha de entrega con su
incertidumbre). Predice el TIEMPO RESTANTE (= D - día actual) en 5 cuantiles.

Reporta: cobertura empírica de los intervalos (¿el 80% real cae dentro del [p10,p90]?),
ancho del intervalo por etapa, calibración de cuantiles, y bandas para cosechas de
ejemplo. Validación GroupKFold por cosecha (out-of-fold).
"""
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
from sklearn.model_selection import GroupKFold
from lightgbm import LGBMRegressor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG = 42
L = pd.read_parquet("landmarks_clima.parquet")
L["restante"] = L["D"] - L["dia_actual"]
FEATS = [c for c in L.columns if c not in ["ID_COSECHA","ID_TANDA","D","restante"]]
y = L["restante"].to_numpy(float)
groups = L["ID_COSECHA"].to_numpy()
QS = [0.1, 0.25, 0.5, 0.75, 0.9]

def modelo(q):
    return LGBMRegressor(objective="quantile", alpha=q, n_estimators=500, learning_rate=0.05,
                         num_leaves=63, min_child_samples=40, subsample=0.8, subsample_freq=1,
                         colsample_bytree=0.7, reg_lambda=3, random_state=RNG, n_jobs=-1, verbose=-1)

oof = {q: np.zeros(len(L)) for q in QS}
for tr, te in GroupKFold(5).split(L, y, groups):
    for q in QS:
        oof[q][te] = modelo(q).fit(L.iloc[tr][FEATS], y[tr]).predict(L.iloc[te][FEATS])
# evitar cruce de cuantiles: ordenar por fila
Q = np.sort(np.column_stack([oof[q] for q in QS]), axis=1)
p10, p25, p50, p75, p90 = Q.T
for name, arr in zip(["p10","p25","p50","p75","p90"], [p10,p25,p50,p75,p90]):
    L[name] = arr

# --- cobertura y ancho ---
def pinball(t, pred, q): return np.mean(np.maximum(q*(t-pred), (q-1)*(t-pred)))
cov80 = np.mean((y>=p10)&(y<=p90)); cov50 = np.mean((y>=p25)&(y<=p75))
print("=== INTERVALOS DE PREDICCIÓN (tiempo restante, días) ===")
print(f"Cobertura [p10,p90] (objetivo 80%): {cov80*100:.1f}%")
print(f"Cobertura [p25,p75] (objetivo 50%): {cov50*100:.1f}%")
print(f"Ancho medio [p10,p90]: {np.mean(p90-p10):.1f} d | MAE de p50: {np.mean(np.abs(y-p50)):.2f} d")
print("\nCobertura y ancho por etapa del ciclo:")
print(f"  {'etapa':10s} {'n':>6s} {'cob80':>7s} {'ancho80':>8s} {'MAE p50':>8s}")
buckets=[(3,10),(11,20),(21,30),(31,45),(46,100)]
for lo,hi in buckets:
    m=((L['dia_actual']>=lo)&(L['dia_actual']<=hi)).values
    if m.sum()==0: continue
    print(f"  {f'{lo}-{hi}':10s} {int(m.sum()):6d} {np.mean((y[m]>=p10[m])&(y[m]<=p90[m]))*100:6.1f}% "
          f"{np.mean(p90[m]-p10[m]):7.1f}d {np.mean(np.abs(y[m]-p50[m])):7.2f}d")

# --- FIG 1: calibración de cuantiles + ancho por etapa ---
fig,ax=plt.subplots(1,2,figsize=(13,5))
emp=[np.mean(y<=L[n].values) for n in ["p10","p25","p50","p75","p90"]]
ax[0].plot([0,1],[0,1],"k--",lw=1,label="calibración perfecta")
ax[0].plot(QS,emp,"o-",color="#2f7de8",ms=8)
for q,e in zip(QS,emp): ax[0].annotate(f"{e*100:.0f}%",(q,e),textcoords="offset points",xytext=(6,-10),fontsize=8)
ax[0].set_xlabel("cuantil nominal"); ax[0].set_ylabel("cobertura empírica P(real ≤ predicho)")
ax[0].set_title("Calibración de los cuantiles"); ax[0].grid(alpha=0.3); ax[0].legend()
xa=[(lo+hi)/2 for lo,hi in buckets]
w=[np.mean(p90[((L['dia_actual']>=lo)&(L['dia_actual']<=hi)).values]-p10[((L['dia_actual']>=lo)&(L['dia_actual']<=hi)).values]) for lo,hi in buckets]
ax[1].plot(xa,w,"o-",color="#d6336c")
ax[1].set_xlabel("día del ciclo en que se predice"); ax[1].set_ylabel("ancho del intervalo p10–p90 (días)")
ax[1].set_title("La incertidumbre se achica al avanzar el ciclo"); ax[1].grid(alpha=0.3)
fig.tight_layout(); fig.savefig("int_calibracion.png",dpi=120)

# --- FIG 2: bandas de predicción para cosechas de ejemplo ---
L["key"]=L["ID_COSECHA"]+"·t"+L["ID_TANDA"].astype(str)
durs=L.groupby("key")["D"].first()
sel=[]
for lo,hi in [(30,38),(43,49),(55,95)]:
    c=durs[(durs>=lo)&(durs<=hi)].index
    sel+=list(c[:2])
fig,axes=plt.subplots(2,3,figsize=(15,8))
for ax,key in zip(axes.ravel(),sel[:6]):
    s=L[L["key"]==key].sort_values("dia_actual"); D=s["D"].iloc[0]; d=s["dia_actual"].values
    ax.fill_between(d, d+s["p10"], d+s["p90"], alpha=0.25, color="#2f7de8", label="intervalo 80% (p10–p90)")
    ax.plot(d, d+s["p50"], "-", color="#2f7de8", label="predicción p50 (día de cosecha)")
    ax.axhline(D,color="#d6336c",ls="--",label=f"cosecha real = {D:.0f}")
    ax.set_title(key,fontsize=10); ax.set_xlabel("día del ciclo"); ax.grid(alpha=0.3); ax.legend(fontsize=7)
axes[0,0].set_ylabel("día de cosecha estimado"); axes[1,0].set_ylabel("día de cosecha estimado")
fig.suptitle("Cómo se usa el dato: día de cosecha estimado con su intervalo, actualizado cada día",fontsize=13)
fig.tight_layout(); fig.savefig("int_bandas_cosechas.png",dpi=120)
print("\nFiguras: int_calibracion.png, int_bandas_cosechas.png")
