"""Baseline agronómico: TIEMPO TÉRMICO (grados-día, GDD) vs ML.

Hipótesis clásica de agronomía (muy usada en invernaderos): la cosecha ocurre cuando se
acumula un tiempo térmico ~constante (grados-día por encima de una temperatura base).
Si eso se cumple, un modelo tan simple como "faltan (GDD_objetivo - GDD_acumulado) /
GDD_diario_esperado días" competiría con el ML.

1) ¿Qué temperatura base hace más constante el GDD-a-cosecha? (menor CV)
2) Baseline GDD predictivo (GroupKFold) vs el modelo definitivo (2.80).
"""
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error

ENR = "../dataset_enrichment"
raw = pd.read_csv("../dataset/dataset_albahaca_1.csv", parse_dates=["FECHA"])
raw = raw.sort_values(["ID_COSECHA","ID_TANDA","DIA_DESDE_INICIO"])
era5 = pd.read_csv(f"{ENR}/era5_real/clima_era5_diario.csv", parse_dates=["FECHA"]).set_index("FECHA")
clim = pd.read_csv(f"{ENR}/climatologia/climatologia_doy.csv").set_index("doy")
tmean = era5["temperature_2m_mean"]

# tandas completas: secuencia de fechas + día de cosecha
tandas=[]
for (cos,tanda),g in raw.groupby(["ID_COSECHA","ID_TANDA"]):
    fin=g[g["COSECHA_FIN"]==1]
    if len(fin)==0: continue
    D=fin["DIA_DESDE_INICIO"].min()
    g=g[g["DIA_DESDE_INICIO"]<=D].sort_values("DIA_DESDE_INICIO")
    fechas=pd.DatetimeIndex(g["FECHA"].values)
    tandas.append({"cos":cos,"tanda":tanda,"D":D,"mes":fechas[0].month,"fechas":fechas})

def gdd_series(base):
    return (tmean-base).clip(lower=0)

# 1) ¿qué base minimiza la variabilidad del GDD-a-cosecha?
print("=== GDD acumulado a cosecha, por temperatura base ===")
print(f"  {'base':>5s} {'GDD medio':>10s} {'CV%':>7s}  (CV bajo = tiempo térmico consistente)")
best=None
for base in [2,4,6,8,10]:
    gd=gdd_series(base)
    acc=[gd.reindex(t["fechas"]).sum() for t in tandas]
    acc=np.array(acc); cv=np.nanstd(acc)/np.nanmean(acc)*100
    print(f"  {base:5d} {np.nanmean(acc):10.1f} {cv:6.1f}%")
    if best is None or cv<best[1]: best=(base,cv)
BASE=best[0]
print(f"-> base elegida: {BASE} °C (CV {best[1]:.1f}%)")

# 2) baseline GDD predictivo, GroupKFold
gd=gdd_series(BASE)
clim_gdd_doy=(clim["temperature_2m_mean"]-BASE).clip(lower=0)  # GDD diario esperado por día del año
# landmarks: por tanda y día d, predecir D
rows=[]
for t in tandas:
    fechas=t["fechas"]; gacc=gd.reindex(fechas).fillna(0).cumsum().values
    for i,d in enumerate(range(1,t["D"]+1)):
        if d<3 or d>=t["D"]: continue
        # GDD diario esperado en los próximos días (climatología por doy)
        prox=pd.date_range(fechas[i]+pd.Timedelta(days=1),periods=14,freq="D")
        rate=clim_gdd_doy.reindex(prox.dayofyear).mean()
        rows.append({"cos":t["cos"],"D":t["D"],"dia":d,"gdd_acc":gacc[i],
                     "rate":max(rate,0.3)})  # floor para no explotar en invierno
Ld=pd.DataFrame(rows)
y=Ld["D"].to_numpy(float); groups=Ld["cos"].values
oof=np.zeros(len(Ld))
for tr,te in GroupKFold(5).split(Ld,y,groups):
    gdd_target=np.median([t["D"] for t in tandas])  # placeholder
    # objetivo = mediana del GDD-a-cosecha en train
    tr_cos=set(groups[tr])
    gtarget=np.median([gd.reindex(t["fechas"]).sum() for t in tandas if t["cos"] in tr_cos])
    rem=(gtarget-Ld.iloc[te]["gdd_acc"].values)/Ld.iloc[te]["rate"].values
    rem=np.clip(rem,0,120)
    oof[te]=Ld.iloc[te]["dia"].values+rem
mae=mean_absolute_error(y,oof)
print(f"\n=== Baseline GDD predictivo (base {BASE}) ===")
print(f"  MAE = {mae:.2f} d   [modelo definitivo ML = 2.80 d | dummy = 8.19]")
for lo,hi in [(3,15),(15,100)]:
    m=(Ld['dia'].values>=lo)&(Ld['dia'].values<hi)
    if m.sum(): print(f"    día {lo}-{hi}: MAE={mean_absolute_error(y[m],oof[m]):.2f}")
