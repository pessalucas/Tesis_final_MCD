"""Dispersión de la duración predicha por cada modelo vs. la dispersión REAL.

Muestra la 'reversión a la media': los modelos comprimen el rango de duraciones.
Compara 4 modelos (dummy, estáticas, completo, +clima) contra la duración real,
en etapa temprana (día 3-10) y tardía (día 30+), con predicciones out-of-fold.
"""
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore", category=RuntimeWarning)
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG = 42; MIN_DIA = 3
DS = "../dataset/dataset_albahaca_1.csv"; ENR = "../dataset_enrichment"
EVENTOS = ["TRASPASO_TANQUE","AGREGADO_SALES","CAMBIO_TANQUE","CAMBIO_80_TANQUE","BOVEL","ESTEN_80","G_A_F1"]

df = pd.read_csv(DS, parse_dates=["FECHA"])
df["PPM"] = np.where(df["UNIDAD"].eq("mS/cm"), df["PPM"]*500, df["PPM"])
df.loc[(df["PH"]<0)|(df["PH"]>14),"PH"]=np.nan
df = df.sort_values(["ID_COSECHA","ID_TANDA","DIA_DESDE_INICIO"]).reset_index(drop=True)

def info(sub):
    fin=sub[sub["COSECHA_FIN"]==1]
    return pd.Series({"duracion":fin["DIA_DESDE_INICIO"].min() if len(fin) else np.nan,
                      "cultivo":sub["CULTIVO"].iloc[0],"fraccion":str(sub["FRACCION"].iloc[0]),
                      "mes_inicio":sub["FECHA"].min().month})
comp = df.groupby(["ID_COSECHA","ID_TANDA"]).apply(info, include_groups=False)
comp = comp[comp["duracion"].notna()]
cult_cod={c:i for i,c in enumerate(sorted(df["CULTIVO"].dropna().unique()))}
frac_cod={f:i for i,f in enumerate(sorted(df["FRACCION"].astype(str).unique()))}

# enrichment clima
era5=pd.read_csv(f"{ENR}/era5_real/clima_era5_diario.csv",parse_dates=["FECHA"]).set_index("FECHA")
clim=pd.read_csv(f"{ENR}/climatologia/climatologia_doy.csv").set_index("doy")
pron=pd.read_csv(f"{ENR}/pronostico_lead/pronostico_lead_por_emision.csv",parse_dates=["FECHA"]).set_index("FECHA")
rango=pd.date_range(df["FECHA"].min()-pd.Timedelta(days=70),df["FECHA"].max()+pd.Timedelta(days=90),freq="D")
exp_dli=pd.Series(clim["dli_mol_m2"].reindex(rango.dayofyear).values,index=rango).cumsum()
cd_dli=pd.Series(clim["dli_mol_m2"].reindex(rango.dayofyear).values,index=rango)
e_temp,e_dli,e_gdd=era5["temperature_2m_mean"],era5["dli_mol_m2"],era5["gdd_base10"]

def slope(y):
    y=np.asarray(y,float);m=~np.isnan(y)
    return np.nan if m.sum()<2 else np.polyfit(np.arange(len(y))[m],y[m],1)[0]
def dd(ev,dias,i):
    oc=np.where(np.nan_to_num(ev[:i+1])==1)[0]
    return np.nan if len(oc)==0 else float(dias[i]-dias[oc[-1]])
def futm(cs,F,n):
    try: return (cs.loc[F+pd.Timedelta(days=n)]-cs.loc[F])/n
    except KeyError: return np.nan

rows=[]
for (cos,tanda),g in df.groupby(["ID_COSECHA","ID_TANDA"]):
    if (cos,tanda) not in comp.index: continue
    D=comp.loc[(cos,tanda),"duracion"];mes=comp.loc[(cos,tanda),"mes_inicio"]
    cult=comp.loc[(cos,tanda),"cultivo"];frac=comp.loc[(cos,tanda),"fraccion"]
    g=g.sort_values("DIA_DESDE_INICIO")
    ph,ppm=g["PH"].to_numpy(float),g["PPM"].to_numpy(float)
    dias=g["DIA_DESDE_INICIO"].to_numpy();tanque=g["ID_TANQUE"].to_numpy()
    fechas=pd.DatetimeIndex(g["FECHA"].to_numpy());F0=fechas[0]
    ev={e:g[e].to_numpy() for e in EVENTOS}
    ct=e_temp.reindex(fechas).to_numpy();cdli=e_dli.reindex(fechas).to_numpy();cgdd=e_gdd.reindex(fechas).to_numpy()
    kdli=cd_dli.reindex(fechas).to_numpy()
    prev=pd.date_range(F0-pd.Timedelta(days=60),F0-pd.Timedelta(days=1),freq="D")
    p60dli=e_dli.reindex(prev).mean()
    for i,d in enumerate(dias):
        if d<MIN_DIA or d>=D: continue
        hph,hppm=ph[:i+1],ppm[:i+1];w=slice(max(0,i-6),i+1);F=fechas[i]
        rows.append({"ID_COSECHA":cos,"key":f"{cos}·t{tanda}","D":D,"dia_actual":d,
            "mes_inicio":mes,"mes_sin":np.sin(2*np.pi*mes/12),"mes_cos":np.cos(2*np.pi*mes/12),
            "cultivo_cod":cult_cod.get(cult,-1),"fraccion_cod":frac_cod.get(str(frac),-1),
            "ph_last":hph[-1],"ph_mean":np.nanmean(hph),"ph_std":np.nanstd(hph),"ph_min":np.nanmin(hph),
            "ph_max":np.nanmax(hph),"ph_slope":slope(hph),"ph_mean7":np.nanmean(ph[w]),
            "ppm_last":hppm[-1],"ppm_mean":np.nanmean(hppm),"ppm_std":np.nanstd(hppm),"ppm_min":np.nanmin(hppm),
            "ppm_max":np.nanmax(hppm),"ppm_slope":slope(hppm),"ppm_mean7":np.nanmean(ppm[w]),
            "tiene_tanque":int(not np.isnan(tanque[i])),"dias_en_tanque":int(np.sum(~np.isnan(tanque[:i+1]))),
            **{f"n_{e}":int(np.nansum(ev[e][:i+1])) for e in EVENTOS},
            **{f"dias_desde_{e}":dd(ev[e],dias,i) for e in EVENTOS},
            "cl_dli_mean_ciclo":np.nanmean(cdli[:i+1]),"cl_temp_mean_ciclo":np.nanmean(ct[:i+1]),
            "cl_gdd_acum_ciclo":np.nansum(cgdd[:i+1]),"cl_prev60_dli":p60dli,
            "cl_anom_dli_ciclo":np.nanmean(cdli[:i+1])-np.nanmean(kdli[:i+1]),
            "cl_fut14_dli_clim":futm(exp_dli,F,14),"cl_fc_dli_next7":pron["dli__next7"].get(F,np.nan)})
L=pd.DataFrame(rows)
y=L["D"].to_numpy(float);groups=L["ID_COSECHA"].to_numpy()

ESTAT=["mes_inicio","mes_sin","mes_cos","cultivo_cod","fraccion_cod","dia_actual"]
TRAJ=[c for c in L.columns if c not in ESTAT+["ID_COSECHA","key","D"] and not c.startswith("cl_")]
CLIMA=[c for c in L.columns if c.startswith("cl_")]
SETS={"Estáticas":ESTAT,"Completo":ESTAT+TRAJ,"+Clima":ESTAT+TRAJ+CLIMA}

def nuevo(): return HistGradientBoostingRegressor(max_depth=4,max_iter=300,learning_rate=0.05,
    min_samples_leaf=40,l2_regularization=1.0,random_state=RNG)
gkf=GroupKFold(5)
oof={"Dummy":np.zeros(len(L))}
for k in SETS: oof[k]=np.zeros(len(L))
for tr,te in gkf.split(L,y,groups):
    oof["Dummy"][te]=y[tr].mean()
    for k,feats in SETS.items():
        oof[k][te]=nuevo().fit(L.iloc[tr][feats],y[tr]).predict(L.iloc[te][feats])
for k in oof: L[f"p_{k}"]=oof[k]

MODELOS=["Dummy","Estáticas","Completo","+Clima"]
COLORS=["#999999","#e8952f","#2f7de8","#37b24d"]

def panel(ax, mask, titulo):
    # duración real por tanda (una vez) + predicción media por tanda en la ventana
    real=L.loc[mask].groupby("key")["D"].first().values
    data=[real]+[L.loc[mask].groupby("key")[f"p_{m}"].mean().values for m in MODELOS]
    labels=["REAL"]+MODELOS
    parts=ax.violinplot(data,showmeans=False,showextrema=True)
    for j,pc in enumerate(parts['bodies']):
        pc.set_facecolor((["#d6336c"]+COLORS)[j]); pc.set_alpha(0.55)
    ax.set_xticks(range(1,len(labels)+1)); ax.set_xticklabels(labels)
    ax.set_ylabel("duración total (días)"); ax.set_title(titulo); ax.grid(alpha=0.3,axis="y")
    for j,dvals in enumerate(data):
        ax.text(j+1, ax.get_ylim()[1], f"σ={np.nanstd(dvals):.1f}", ha="center", va="bottom", fontsize=8)

fig,axes=plt.subplots(1,2,figsize=(15,6),sharey=True)
panel(axes[0], (L["dia_actual"]<=10).values, "Estimación TEMPRANA (día 3-10)")
panel(axes[1], (L["dia_actual"]>=30).values, "Estimación TARDÍA (día ≥30)")
fig.suptitle("Dispersión de la duración: cada modelo vs. la REAL (σ = desvío estándar)",fontsize=13)
fig.tight_layout(); fig.savefig("comparar_dispersion.png",dpi=120)

print("=== Desvío estándar de la duración (días) ===")
for etq,mask in [("temprana (3-10)",(L["dia_actual"]<=10).values),("tardía (>=30)",(L["dia_actual"]>=30).values)]:
    real=L.loc[mask].groupby("key")["D"].first()
    print(f"\n  {etq}:  REAL σ={real.std():.2f}")
    for m in MODELOS:
        s=L.loc[mask].groupby("key")[f"p_{m}"].mean().std()
        print(f"      {m:10s} σ={s:.2f}  ({100*s/real.std():.0f}% de la real)")
print("\nGráfico: comparar_dispersion.png")
