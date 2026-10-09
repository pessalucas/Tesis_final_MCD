"""¿Vale la pena un pronóstico estacional? Test del ORÁCULO (upper bound).

Le damos al modelo el clima REAL del resto del ciclo (días d+1..cosecha) como features:
anomalías de temperatura, luz (DLI) y precipitación respecto de lo normal. Eso es un
pronóstico PERFECTO (imposible en producción -> leakage a propósito). Si ni siquiera el
oráculo baja el MAE sobre el campeón (2.80), ningún pronóstico estacional real va a servir.

Además: oráculo solo del tramo LEJANO (día d+8 en adelante), que es lo que un estacional
agregaría sobre el pronóstico de 7 días que ya tenemos.
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
raw=pd.read_csv("../dataset/dataset_albahaca_1.csv",parse_dates=["FECHA"]).sort_values(["ID_COSECHA","ID_TANDA","DIA_DESDE_INICIO"])
era5=pd.read_csv("../dataset_enrichment/era5_real/clima_era5_diario.csv",parse_dates=["FECHA"]).set_index("FECHA")
clim=pd.read_csv("../dataset_enrichment/climatologia/climatologia_doy.csv").set_index("doy")
rango=pd.date_range(era5.index.min()-pd.Timedelta(days=5),era5.index.max()+pd.Timedelta(days=5),freq="D")
cd={c:pd.Series(clim[s].reindex(rango.dayofyear).values,index=rango) for c,s in
    [("temp","temperature_2m_mean"),("dli","dli_mol_m2"),("prec","precipitation_sum")]}
er={"temp":era5["temperature_2m_mean"],"dli":era5["dli_mol_m2"],"prec":era5["precipitation_sum"]}

# fechas por (cosecha,tanda) y día
key2fechas={}
for (cos,tanda),g in raw.groupby(["ID_COSECHA","ID_TANDA"]):
    key2fechas[(cos,tanda)]=pd.DatetimeIndex(g.sort_values("DIA_DESDE_INICIO")["FECHA"].values)

def anom_rem(row, desde_offset):
    fechas=key2fechas.get((row["ID_COSECHA"],row["ID_TANDA"]))
    d=int(row["dia_actual"]); D=int(row["D"])
    # días restantes desde d+desde_offset hasta D (índices 0-based: día k -> fechas[k-1])
    ini=d+desde_offset
    if ini>=D or fechas is None or len(fechas)<D: return (np.nan,np.nan,np.nan)
    rem=fechas[ini-1:D-1]  # fechas de los días ini..D-1
    if len(rem)==0: return (np.nan,np.nan,np.nan)
    at=(er["temp"].reindex(rem)-cd["temp"].reindex(rem)).mean()
    ad=(er["dli"].reindex(rem)-cd["dli"].reindex(rem)).mean()
    ap=(er["prec"].reindex(rem)-cd["prec"].reindex(rem)).mean()
    return (at,ad,ap)

# oráculo total (desde d+1) y lejano (desde d+8)
for off,pref in [(1,"orc"),(8,"orclejano")]:
    vals=L.apply(lambda r: anom_rem(r,off),axis=1,result_type="expand")
    L[f"{pref}_temp"],L[f"{pref}_dli"],L[f"{pref}_prec"]=vals[0],vals[1],vals[2]

FEATS=[c for c in L.columns if c not in ["ID_COSECHA","ID_TANDA","D"] and not c.startswith(("orc","orclejano"))]
ORC=[c for c in L.columns if c.startswith("orc_")]
ORCLEJ=[c for c in L.columns if c.startswith("orclejano_")]
y=L["D"].to_numpy(float);groups=L["ID_COSECHA"].to_numpy()
LGB=dict(objective="regression_l1",n_estimators=800,learning_rate=0.03,num_leaves=63,max_depth=-1,
    min_child_samples=20,subsample=0.7,subsample_freq=1,colsample_bytree=0.6,reg_lambda=5,random_state=RNG,n_jobs=-1,verbose=-1)
SETS={"campeón (sin estacional)":FEATS,"+ oráculo TOTAL (d+1..fin)":FEATS+ORC,
      "+ oráculo LEJANO (d+8..fin)":FEATS+ORCLEJ}
oof={k:np.zeros(len(L)) for k in SETS}
for tr,te in GroupKFold(5).split(L,y,groups):
    for k,f in SETS.items():
        oof[k][te]=LGBMRegressor(**LGB).fit(L.iloc[tr][f],y[tr]).predict(L.iloc[te][f])
print("=== ORÁCULO estacional (upper bound; clima real futuro = pronóstico perfecto) ===")
for k in SETS:
    print(f"  {k:30s} MAE={mean_absolute_error(y,oof[k]):.3f} d")
print("\nSi el oráculo no baja el MAE -> ningún pronóstico estacional real sirve.")
