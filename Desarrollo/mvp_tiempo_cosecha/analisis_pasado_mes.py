"""¿La info PASADA de cuánto demoraron cosechas anteriores es útil? ¿por mes del año?

Dos preguntas de Lucas:
  1. Features CAUSALES de historia (promedio de duración de tandas ya cerradas antes de
     que la actual arranque, del mismo mes / recientes / global) -> ¿cuánto aportan?
  2. Análisis por mes de inicio: variabilidad, predictibilidad, utilidad de la historia.

Todo out-of-fold, GroupKFold por cosecha. Genera gráficos.
"""
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG = 42
L = pd.read_parquet("landmarks_clima.parquet")
raw = pd.read_csv("../dataset/dataset_albahaca_1.csv", parse_dates=["FECHA"])

# --- features causales de historia, a nivel tanda ---
starts = raw.groupby(["ID_COSECHA","ID_TANDA"])["FECHA"].min()
tm = L.drop_duplicates(["ID_COSECHA","ID_TANDA"]).set_index(["ID_COSECHA","ID_TANDA"])[["D","mes_inicio"]].copy()
tm["start"] = starts.reindex(tm.index)
tm["end"] = tm["start"] + pd.to_timedelta(tm["D"], unit="D")
tm = tm.sort_values("start")
glob = tm["D"].mean()

feats_hist = []
for idx, row in tm.iterrows():
    ya = tm[tm["end"] < row["start"]]                       # tandas cerradas antes del inicio
    mm = ya[ya["mes_inicio"] == row["mes_inicio"]]          # mismo mes de inicio
    rec = ya.sort_values("end").tail(5)                     # 5 más recientes
    feats_hist.append({"ID_COSECHA": idx[0], "ID_TANDA": idx[1],
        "hist_dur_mismomes": mm["D"].mean() if len(mm) else (ya["D"].mean() if len(ya) else glob),
        "hist_n_mismomes": len(mm),
        "hist_dur_reciente5": rec["D"].mean() if len(rec) else glob,
        "hist_dur_global": ya["D"].mean() if len(ya) else glob})
H = pd.DataFrame(feats_hist)
L = L.merge(H, on=["ID_COSECHA","ID_TANDA"], how="left")

BASE = [c for c in L.columns if c not in ["ID_COSECHA","ID_TANDA","D"] and not c.startswith("hist_")]
HIST = [c for c in L.columns if c.startswith("hist_")]
y = L["D"].to_numpy(float); groups = L["ID_COSECHA"].to_numpy()
def hgb(): return HistGradientBoostingRegressor(max_depth=4,max_iter=300,learning_rate=0.05,
    min_samples_leaf=40,l2_regularization=1.0,random_state=RNG)

SETS = {"solo historia (+día)": HIST+["dia_actual"], "base (sin historia)": BASE,
        "base + historia": BASE+HIST}
oof = {k: np.zeros(len(L)) for k in SETS}
for tr,te in GroupKFold(5).split(L,y,groups):
    for k,f in SETS.items():
        oof[k][te] = hgb().fit(L.iloc[tr][f], y[tr]).predict(L.iloc[te][f])
print("=== ¿Aporta la historia de duraciones? (MAE OOF, días) ===")
for k in SETS:
    print(f"  {k:24s} MAE={mean_absolute_error(y,oof[k]):.2f} d")

# importancia de features de historia dentro del modelo completo
from sklearn.inspection import permutation_importance
m=hgb().fit(L[BASE+HIST],y)
pi=permutation_importance(m,L[BASE+HIST],y,n_repeats=4,random_state=RNG,scoring="neg_mean_absolute_error")
imp=pd.Series(pi.importances_mean,index=BASE+HIST).sort_values(ascending=False)
print("\nRanking de las features de historia (posición y valor):")
for h in HIST:
    print(f"  {h:22s} imp={imp[h]:.3f}  (puesto {list(imp.index).index(h)+1} de {len(imp)})")

# --- análisis por mes ---
L["pred_full"]=oof["base + historia"]; L["ae"]=np.abs(L["D"]-L["pred_full"])
tanda_lv=L.drop_duplicates(["ID_COSECHA","ID_TANDA"])
por_mes=L.groupby("mes_inicio").agg(n_land=("D","size"),dur_media=("D","mean"),
    dur_sd=("D","std"),MAE=("ae","mean")).round(2)
print("\n=== Por mes de inicio ===")
print(por_mes)

MESES=["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"]
fig,ax=plt.subplots(1,3,figsize=(18,5))
# 1) distribución de duración por mes (a nivel tanda)
data=[tanda_lv[tanda_lv["mes_inicio"]==m]["D"].values for m in range(1,13)]
bp=ax[0].boxplot(data,tick_labels=MESES,patch_artist=True)
for p in bp['boxes']: p.set_facecolor("#74c0fc")
ax[0].set_title("Duración de la cosecha por mes de inicio"); ax[0].set_ylabel("días"); ax[0].grid(alpha=0.3,axis="y")
# 2) MAE por mes
mm=por_mes.reindex(range(1,13))
ax[1].bar(MESES,mm["MAE"],color="#2f7de8"); ax[1].set_title("Error del modelo por mes de inicio")
ax[1].set_ylabel("MAE (días)"); ax[1].grid(alpha=0.3,axis="y")
# 3) variabilidad (sd) por mes
ax[2].bar(MESES,mm["dur_sd"],color="#d6336c"); ax[2].set_title("Variabilidad de la duración por mes (sd)")
ax[2].set_ylabel("desvío estándar (días)"); ax[2].grid(alpha=0.3,axis="y")
fig.suptitle("Análisis por mes: duración, error y variabilidad",fontsize=13)
fig.tight_layout(); fig.savefig("mes_analisis.png",dpi=120)
print("\nFigura: mes_analisis.png")
