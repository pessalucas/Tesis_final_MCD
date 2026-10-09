"""¿El futuro de QUÉ feature aportaría valor? (oráculo por-feature)

Para cada variable dinámica, le damos al modelo el resumen REAL de su trayectoria en el
tramo restante del ciclo (días d+1..cosecha) = un pronóstico perfecto de ESA feature.
Medimos el lift marginal sobre el campeón. Si el futuro de alguna baja el MAE, vale la
pena intentar forecastearla; si no, no.

Se usan resúmenes de NIVEL (media/último/pendiente), no conteos, para no filtrar la
longitud restante (que es el target).
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
raw["PPM"]=np.where(raw["UNIDAD"].eq("mS/cm"),raw["PPM"]*500,raw["PPM"])
raw.loc[(raw["PH"]<0)|(raw["PH"]>14),"PH"]=np.nan
era5=pd.read_csv("../dataset_enrichment/era5_real/clima_era5_diario.csv",parse_dates=["FECHA"]).set_index("FECHA")

# arrays por tanda
T={}
for (cos,tanda),g in raw.groupby(["ID_COSECHA","ID_TANDA"]):
    g=g.sort_values("DIA_DESDE_INICIO")
    T[(cos,tanda)]=dict(dias=g["DIA_DESDE_INICIO"].to_numpy(),ph=g["PH"].to_numpy(float),
        ppm=g["PPM"].to_numpy(float),fechas=pd.DatetimeIndex(g["FECHA"].values))
edli=era5["dli_mol_m2"]; etemp=era5["temperature_2m_mean"]

def slope(a):
    a=np.asarray(a,float);m=~np.isnan(a)
    return np.nan if m.sum()<2 else np.polyfit(np.arange(len(a))[m],a[m],1)[0]

def oracle_row(r):
    t=T.get((r["ID_COSECHA"],r["ID_TANDA"]));d=int(r["dia_actual"]);D=int(r["D"])
    if t is None: return [np.nan]*8
    dias=t["dias"]; rem=(dias>d)&(dias<=D)
    if rem.sum()==0: return [np.nan]*8
    ph=t["ph"][rem]; ppm=t["ppm"][rem]; fr=t["fechas"][rem]
    dli=edli.reindex(fr).to_numpy(); tp=etemp.reindex(fr).to_numpy()
    return [np.nanmean(ph),ph[-1] if len(ph) else np.nan,slope(ph),
            np.nanmean(ppm),ppm[-1] if len(ppm) else np.nan,slope(ppm),
            np.nanmean(dli),np.nanmean(tp)]

cols=["of_ph_mean","of_ph_last","of_ph_slope","of_ppm_mean","of_ppm_last","of_ppm_slope","of_dli_mean","of_temp_mean"]
L[cols]=L.apply(oracle_row,axis=1,result_type="expand")

BASE=[c for c in L.columns if c not in ["ID_COSECHA","ID_TANDA","D"] and not c.startswith("of_")]
y=L["D"].to_numpy(float);groups=L["ID_COSECHA"].to_numpy()
LGB=dict(objective="regression_l1",n_estimators=800,learning_rate=0.03,num_leaves=63,max_depth=-1,
    min_child_samples=20,subsample=0.7,subsample_freq=1,colsample_bytree=0.6,reg_lambda=5,random_state=RNG,n_jobs=-1,verbose=-1)
def cv(feats):
    oof=np.zeros(len(L))
    for tr,te in GroupKFold(5).split(L,y,groups):
        oof[te]=LGBMRegressor(**LGB).fit(L.iloc[tr][feats],y[tr]).predict(L.iloc[te][feats])
    return mean_absolute_error(y,oof)

base=cv(BASE)
print(f"=== Oráculo por-feature: lift de conocer el FUTURO de cada variable ===")
print(f"  campeón (sin oráculo)          MAE={base:.3f}")
GRUPOS={"futuro pH":["of_ph_mean","of_ph_last","of_ph_slope"],
        "futuro PPM":["of_ppm_mean","of_ppm_last","of_ppm_slope"],
        "futuro luz (DLI)":["of_dli_mean"],
        "futuro temperatura":["of_temp_mean"],
        "futuro pH+PPM (química)":["of_ph_mean","of_ph_last","of_ph_slope","of_ppm_mean","of_ppm_last","of_ppm_slope"],
        "futuro TODO":cols}
for etq,g in GRUPOS.items():
    m=cv(BASE+g)
    print(f"  + {etq:28s} MAE={m:.3f}  (Δ={m-base:+.3f})")
print("\nΔ negativo grande = ese futuro SÍ aportaría -> vale forecastear esa variable.")
