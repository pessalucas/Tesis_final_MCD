"""EXP-A1 · MVP de SUPERVIVENCIA (tiempo-al-evento) con censura.

A diferencia de la regresión (model.py), este framing USA las 62 tandas censuradas
(ciclos abiertos al final del registro): aportan la información "esta tanda llevaba X
días sin cosecharse". Enfoque landmark de supervivencia: en cada día d de cada tanda,
el 'tiempo restante' hasta la cosecha es el evento, censurado para las tandas abiertas.

Modelo: Random Survival Forest (scikit-survival).
Métricas: C-index (usa la censura) + MAE del tiempo restante esperado (RMST) sobre las
tandas cerradas, comparable a la regresión (~3.17 d). Validación GroupKFold por cosecha.
"""
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored

RNG = 42; MIN_DIA = 3
DS = "../dataset/dataset_albahaca_1.csv"
EVENTOS = ["TRASPASO_TANQUE","AGREGADO_SALES","CAMBIO_TANQUE","CAMBIO_80_TANQUE","BOVEL","ESTEN_80","G_A_F1"]

df = pd.read_csv(DS, parse_dates=["FECHA"])
df["PPM"] = np.where(df["UNIDAD"].eq("mS/cm"), df["PPM"]*500, df["PPM"])
df.loc[(df["PH"]<0)|(df["PH"]>14),"PH"]=np.nan
df = df.sort_values(["ID_COSECHA","ID_TANDA","DIA_DESDE_INICIO"]).reset_index(drop=True)
cult_cod={c:i for i,c in enumerate(sorted(df["CULTIVO"].dropna().unique()))}
frac_cod={f:i for i,f in enumerate(sorted(df["FRACCION"].astype(str).unique()))}

def slope(y):
    y=np.asarray(y,float);m=~np.isnan(y)
    return np.nan if m.sum()<2 else np.polyfit(np.arange(len(y))[m],y[m],1)[0]
def dd(ev,dias,i):
    oc=np.where(np.nan_to_num(ev[:i+1])==1)[0]
    return np.nan if len(oc)==0 else float(dias[i]-dias[oc[-1]])

rows=[]
for (cos,tanda),g in df.groupby(["ID_COSECHA","ID_TANDA"]):
    g=g.sort_values("DIA_DESDE_INICIO")
    fin=g[g["COSECHA_FIN"]==1]
    completa = len(fin)>0
    endpoint = fin["DIA_DESDE_INICIO"].min() if completa else g["DIA_DESDE_INICIO"].max()
    if endpoint <= MIN_DIA: continue
    mes=g["FECHA"].min().month; cult=g["CULTIVO"].iloc[0]; frac=str(g["FRACCION"].iloc[0])
    ph,ppm=g["PH"].to_numpy(float),g["PPM"].to_numpy(float)
    dias=g["DIA_DESDE_INICIO"].to_numpy(); tanque=g["ID_TANQUE"].to_numpy()
    ev={e:g[e].to_numpy() for e in EVENTOS}
    for i,d in enumerate(dias):
        if d<MIN_DIA or d>=endpoint: continue
        hph,hppm=ph[:i+1],ppm[:i+1];w=slice(max(0,i-6),i+1)
        rows.append({"ID_COSECHA":cos,"key":f"{cos}·t{tanda}","completa":int(completa),
            "restante":float(endpoint-d),"evento":bool(completa),"dia_actual":d,
            "mes_inicio":mes,"mes_sin":np.sin(2*np.pi*mes/12),"mes_cos":np.cos(2*np.pi*mes/12),
            "cultivo_cod":cult_cod.get(cult,-1),"fraccion_cod":frac_cod.get(frac,-1),
            "ph_last":hph[-1],"ph_mean":np.nanmean(hph),"ph_std":np.nanstd(hph),"ph_min":np.nanmin(hph),
            "ph_max":np.nanmax(hph),"ph_slope":slope(hph),"ph_mean7":np.nanmean(ph[w]),
            "ppm_last":hppm[-1],"ppm_mean":np.nanmean(hppm),"ppm_std":np.nanstd(hppm),"ppm_min":np.nanmin(hppm),
            "ppm_max":np.nanmax(hppm),"ppm_slope":slope(hppm),"ppm_mean7":np.nanmean(ppm[w]),
            "tiene_tanque":int(not np.isnan(tanque[i])),"dias_en_tanque":int(np.sum(~np.isnan(tanque[:i+1]))),
            **{f"n_{e}":int(np.nansum(ev[e][:i+1])) for e in EVENTOS},
            **{f"dias_desde_{e}":dd(ev[e],dias,i) for e in EVENTOS}})
L=pd.DataFrame(rows)
FEATS=[c for c in L.columns if c not in ["ID_COSECHA","key","completa","restante","evento"]]
# RSF no acepta NaN -> imputar por mediana (simple, suficiente para MVP)
X=L[FEATS].fillna(L[FEATS].median())
ysurv=Surv.from_arrays(event=L["evento"].values, time=L["restante"].values)
groups=L["ID_COSECHA"].values
print(f"Landmarks {len(L)} | cerrados {int(L['evento'].sum())} | censurados {int((~L['evento']).sum())}")
print(f"Tandas: {L['key'].nunique()} ({L[L['completa']==1]['key'].nunique()} cerradas + {L[L['completa']==0]['key'].nunique()} censuradas)")

cidx_list=[]; mae_surv=np.full(len(L),np.nan); mae_covered=np.zeros(len(L),bool)
for tr,te in GroupKFold(5).split(X,ysurv,groups):
    rsf=RandomSurvivalForest(n_estimators=150,min_samples_leaf=40,max_features="sqrt",
                             n_jobs=-1,random_state=RNG).fit(X.iloc[tr],ysurv[tr])
    # C-index en test (usa censura)
    risk=rsf.predict(X.iloc[te])
    c=concordance_index_censored(L["evento"].values[te],L["restante"].values[te],risk)[0]
    cidx_list.append(c)
    # tiempo restante esperado (RMST = integral de S(t)) para comparar MAE en cerrados
    S=rsf.predict_survival_function(X.iloc[te],return_array=True)
    t=rsf.unique_times_
    rmst=np.trapezoid(S,t,axis=1)
    mae_surv[te]=rmst; mae_covered[te]=True
print(f"\nC-index (GroupKFold): {np.mean(cidx_list):.3f} ± {np.std(cidx_list):.3f}  (0.5=azar, 1=perfecto)")

# MAE del tiempo restante sobre landmarks de tandas CERRADAS (comparable a regresión)
cer=(L["evento"].values)&mae_covered
mae=mean_absolute_error(L["restante"].values[cer],mae_surv[cer])
print(f"MAE tiempo restante (tandas cerradas): {mae:.2f} d   [regresión MVP ~3.17 d]")
# desglose en cola larga (donde la regresión sub-estimaba)
for lo,hi,etq in [(0,15,"restante corto (<15d)"),(15,100,"restante largo (>=15d)")]:
    m=cer&(L["restante"].values>=lo)&(L["restante"].values<hi)
    if m.sum(): print(f"    {etq:26s} n={int(m.sum()):5d}  MAE={mean_absolute_error(L['restante'].values[m],mae_surv[m]):.2f} d")
