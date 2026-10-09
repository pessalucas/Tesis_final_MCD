"""Exploración del MVP: comportamiento dinámico, errores, curva de aprendizaje, cortes.

Usa el modelo MVP (trayectoria, sin clima) con predicciones OUT-OF-FOLD (GroupKFold por
cosecha) para no hacer trampa. Genera 4 figuras + tablas.
"""
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore", category=RuntimeWarning)
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG = 42; MIN_DIA = 3
DS = "../dataset/dataset_albahaca_1.csv"
EVENTOS = ["TRASPASO_TANQUE","AGREGADO_SALES","CAMBIO_TANQUE","CAMBIO_80_TANQUE","BOVEL","ESTEN_80","G_A_F1"]

df = pd.read_csv(DS, parse_dates=["FECHA"])
df["PPM"] = np.where(df["UNIDAD"].eq("mS/cm"), df["PPM"]*500, df["PPM"])
df.loc[(df["PH"]<0)|(df["PH"]>14),"PH"]=np.nan
df = df.sort_values(["ID_COSECHA","ID_TANDA","DIA_DESDE_INICIO"]).reset_index(drop=True)

def info(sub):
    fin = sub[sub["COSECHA_FIN"]==1]
    return pd.Series({"duracion": fin["DIA_DESDE_INICIO"].min() if len(fin) else np.nan,
                      "cultivo": sub["CULTIVO"].iloc[0], "fraccion": str(sub["FRACCION"].iloc[0]),
                      "mes_inicio": sub["FECHA"].min().month})
meta = df.groupby(["ID_COSECHA","ID_TANDA"]).apply(info, include_groups=False)
comp = meta[meta["duracion"].notna()].copy()
cult_cod = {c:i for i,c in enumerate(sorted(df["CULTIVO"].dropna().unique()))}
frac_cod = {f:i for i,f in enumerate(sorted(df["FRACCION"].astype(str).unique()))}

def slope(y):
    y=np.asarray(y,float); m=~np.isnan(y)
    return np.nan if m.sum()<2 else np.polyfit(np.arange(len(y))[m],y[m],1)[0]
def dias_desde(ev,dias,i):
    oc=np.where(np.nan_to_num(ev[:i+1])==1)[0]
    return np.nan if len(oc)==0 else float(dias[i]-dias[oc[-1]])

rows=[]
for (cos,tanda),g in df.groupby(["ID_COSECHA","ID_TANDA"]):
    if (cos,tanda) not in comp.index: continue
    D=comp.loc[(cos,tanda),"duracion"]; mes=comp.loc[(cos,tanda),"mes_inicio"]
    cult=comp.loc[(cos,tanda),"cultivo"]; frac=comp.loc[(cos,tanda),"fraccion"]
    g=g.sort_values("DIA_DESDE_INICIO")
    ph,ppm=g["PH"].to_numpy(float),g["PPM"].to_numpy(float)
    dias=g["DIA_DESDE_INICIO"].to_numpy(); tanque=g["ID_TANQUE"].to_numpy()
    ev={e:g[e].to_numpy() for e in EVENTOS}
    for i,d in enumerate(dias):
        if d<MIN_DIA or d>=D: continue
        hph,hppm=ph[:i+1],ppm[:i+1]; w=slice(max(0,i-6),i+1)
        rows.append({"ID_COSECHA":cos,"ID_TANDA":tanda,"D":D,"dia_actual":d,
            "cultivo":cult,"mes_inicio":mes,
            "mes_sin":np.sin(2*np.pi*mes/12),"mes_cos":np.cos(2*np.pi*mes/12),
            "cultivo_cod":cult_cod.get(cult,-1),"fraccion_cod":frac_cod.get(str(frac),-1),
            "ph_last":hph[-1],"ph_mean":np.nanmean(hph),"ph_std":np.nanstd(hph),"ph_min":np.nanmin(hph),
            "ph_max":np.nanmax(hph),"ph_slope":slope(hph),"ph_mean7":np.nanmean(ph[w]),
            "ppm_last":hppm[-1],"ppm_mean":np.nanmean(hppm),"ppm_std":np.nanstd(hppm),"ppm_min":np.nanmin(hppm),
            "ppm_max":np.nanmax(hppm),"ppm_slope":slope(hppm),"ppm_mean7":np.nanmean(ppm[w]),
            "tiene_tanque":int(not np.isnan(tanque[i])),"dias_en_tanque":int(np.sum(~np.isnan(tanque[:i+1]))),
            **{f"n_{e}":int(np.nansum(ev[e][:i+1])) for e in EVENTOS},
            **{f"dias_desde_{e}":dias_desde(ev[e],dias,i) for e in EVENTOS}})
L=pd.DataFrame(rows)
FEATS=[c for c in L.columns if c not in ["ID_COSECHA","ID_TANDA","D","cultivo"]]
y=L["D"].to_numpy(float); groups=L["ID_COSECHA"].to_numpy()
def nuevo(): return HistGradientBoostingRegressor(max_depth=4,max_iter=300,learning_rate=0.05,
    min_samples_leaf=40,l2_regularization=1.0,random_state=RNG)

# OOF
oof=np.zeros(len(L))
for tr,te in GroupKFold(5).split(L,y,groups):
    oof[te]=nuevo().fit(L.iloc[tr][FEATS],y[tr]).predict(L.iloc[te][FEATS])
L["D_pred"]=oof
L["resid"]=L["D_pred"]-L["D"]
print(f"Landmarks {len(L)} | MAE OOF {mean_absolute_error(y,oof):.2f} d")

# ---------- FIG 1: prediccion corrigiendose dia a dia ----------
L["key"]=L["ID_COSECHA"]+"·t"+L["ID_TANDA"].astype(str)
durs=L.groupby("key")["D"].first()
sel=[]
for lo,hi in [(30,38),(43,49),(55,80)]:
    cand=durs[(durs>=lo)&(durs<=hi)].index
    sel+=list(cand[:2])
fig,axes=plt.subplots(2,3,figsize=(15,8),sharey=False)
for ax,key in zip(axes.ravel(),sel[:6]):
    s=L[L["key"]==key].sort_values("dia_actual")
    D=s["D"].iloc[0]
    ax.plot(s["dia_actual"],s["D_pred"],"o-",color="#2f7de8",ms=3,label="predicción de duración total")
    ax.axhline(D,color="#d6336c",ls="--",label=f"real = {D:.0f} d")
    ax.set_title(key,fontsize=10); ax.set_xlabel("día del ciclo"); ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
axes[0,0].set_ylabel("duración total estimada (d)"); axes[1,0].set_ylabel("duración total estimada (d)")
fig.suptitle("El modelo se corrige día a día: estimación de la duración total vs. real",fontsize=13)
fig.tight_layout(); fig.savefig("exp_correccion_diaria.png",dpi=120)

# ---------- FIG 2: analisis de errores ----------
fig,ax=plt.subplots(1,2,figsize=(13,5))
ax[0].hist(L["resid"],bins=60,color="#2f7de8",alpha=0.8)
ax[0].axvline(0,color="k",lw=1); ax[0].set_title(f"Residuos (pred − real)  ·  media {L['resid'].mean():+.2f} d")
ax[0].set_xlabel("error en días"); ax[0].grid(alpha=0.3)
# scatter a nivel tanda (promedio de pred por tanda) para no saturar
pt=L.groupby("key").agg(D=("D","first"),D_pred=("D_pred","mean")).reset_index()
ax[1].scatter(pt["D"],pt["D_pred"],s=12,alpha=0.5,color="#2f7de8")
lim=[pt["D"].min()-2,pt["D"].max()+2]; ax[1].plot(lim,lim,"k--",lw=1)
ax[1].set_xlabel("duración real (d)"); ax[1].set_ylabel("duración predicha media (d)")
ax[1].set_title("Predicho vs. real (por tanda)"); ax[1].grid(alpha=0.3)
fig.tight_layout(); fig.savefig("exp_errores.png",dpi=120)

# ---------- FIG 3: curva de aprendizaje ----------
cos_all=pd.Series(groups).unique()
rng=np.random.default_rng(RNG); rng.shuffle(cos_all)
test_cos=set(cos_all[:40]); pool=[c for c in cos_all if c not in test_cos]
te_mask=L["ID_COSECHA"].isin(test_cos).values
curve=[]
for n in [20,40,60,80,100,120,146]:
    tr_cos=set(pool[:n]); tr_mask=L["ID_COSECHA"].isin(tr_cos).values
    m=nuevo().fit(L[tr_mask][FEATS],y[tr_mask])
    curve.append((n,mean_absolute_error(y[te_mask],m.predict(L[te_mask][FEATS]))))
curve=np.array(curve)
fig,ax=plt.subplots(figsize=(7,5))
ax.plot(curve[:,0],curve[:,1],"o-",color="#2f7de8")
ax.set_xlabel("nº de cosechas de entrenamiento"); ax.set_ylabel("MAE en test fijo (40 cosechas)")
ax.set_title("Curva de aprendizaje — ¿ganaríamos con más cosechas?"); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig("exp_curva_aprendizaje.png",dpi=120)

# ---------- Tablas: cortes ----------
L["ae"]=L["resid"].abs()
print("\n=== MAE por cultivo ===")
print(L.groupby("cultivo").agg(n_land=("ae","size"),MAE=("ae","mean")).round(2))
print("\n=== MAE por estación de inicio ===")
est={12:"verano",1:"verano",2:"verano",3:"otoño",4:"otoño",5:"otoño",6:"invierno",7:"invierno",8:"invierno",9:"primavera",10:"primavera",11:"primavera"}
L["estacion"]=L["mes_inicio"].map(est)
print(L.groupby("estacion").agg(n=("ae","size"),MAE=("ae","mean"),dur_media=("D","mean")).round(2))
print("\n=== Curva de aprendizaje (MAE vs nº cosechas train) ===")
for n,mae in curve: print(f"  {int(n):3d} cosechas -> MAE {mae:.2f} d")
print("\nFiguras: exp_correccion_diaria.png, exp_errores.png, exp_curva_aprendizaje.png")
