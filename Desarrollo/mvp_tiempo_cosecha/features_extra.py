"""Experimento: ¿aportan valor las features derivadas nuevas?

Calcula, de forma CAUSAL (solo datos hasta el día d, misma grilla que model.py),
familias de features nuevas y las agrega al set del campeón (base+clima+historia).
Mide el aporte por ablación con el mismo LightGBM tuneado:
  - CV agrupada por cosecha (delta fino, comparable con el 2.80)
  - Test temporal (último año) como lectura de producción.

Familias nuevas:
  FOTO   : fotoperíodo (horas de luz por latitud+fecha) — mecanístico, la luz manda.
  VENT   : tendencias/volatilidad por ventana (slopes 3/14 d, std 7/14, aceleración, delta reciente).
  RANGO  : tiempo en rango óptimo de pH y PPM (fracción del ciclo dentro de banda).
  TANQUE : estados de tanque (nº tanques distintos, fracción de días conectado, días pre-tanque).
  EVENT  : tasa de eventos (global y en últimos 7 días).

Correr: uv run features_extra.py
"""
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, median_absolute_error, mean_squared_error
from lightgbm import LGBMRegressor

RNG = 42
CSV = "../dataset/dataset_albahaca_1.csv"
MIN_DIA = 3
LAT = -41.11  # Bariloche
CUT = pd.Timestamp("2025-09-26")
EVENTOS = ["TRASPASO_TANQUE","AGREGADO_SALES","CAMBIO_TANQUE","CAMBIO_80_TANQUE",
           "BOVEL","ESTEN_80","G_A_F1"]
PH_LO, PH_HI = 5.5, 6.5  # rango típico hidropónico

df = pd.read_csv(CSV, parse_dates=["FECHA"])
df["PPM"] = np.where(df["UNIDAD"].eq("mS/cm"), df["PPM"]*500, df["PPM"])
df.loc[(df["PH"]<0)|(df["PH"]>14), "PH"] = np.nan
df = df.sort_values(["ID_COSECHA","ID_TANDA","DIA_DESDE_INICIO"]).reset_index(drop=True)
# banda de PPM data-driven (constante, no depende del target)
PPM_LO, PPM_HI = np.nanpercentile(df["PPM"], [25, 75])

def info_tanda(sub):
    fin = sub[sub["COSECHA_FIN"]==1]
    return pd.Series({"duracion": fin["DIA_DESDE_INICIO"].min() if len(fin) else np.nan})
meta = df.groupby(["ID_COSECHA","ID_TANDA"]).apply(info_tanda, include_groups=False)
completas = meta[meta["duracion"].notna()]

def slope(y):
    y = np.asarray(y, float); m = ~np.isnan(y)
    if m.sum() < 2: return np.nan
    x = np.arange(len(y))[m]
    return np.polyfit(x, y[m], 1)[0]

def daylength(doy):
    lat = np.radians(LAT)
    decl = np.radians(23.45)*np.sin(2*np.pi*(284+doy)/365)
    cosH = np.clip(-np.tan(lat)*np.tan(decl), -1, 1)
    return 24/np.pi*np.arccos(cosH)

def frac_inrango(arr, lo, hi):
    a = np.asarray(arr, float); m = ~np.isnan(a)
    if m.sum() == 0: return np.nan
    return float(np.mean((a[m]>=lo)&(a[m]<=hi)))

rows = []
for (cos, tanda), g in df.groupby(["ID_COSECHA","ID_TANDA"]):
    if (cos, tanda) not in completas.index: continue
    D = completas.loc[(cos, tanda), "duracion"]
    g = g.sort_values("DIA_DESDE_INICIO")
    ph = g["PH"].to_numpy(float); ppm = g["PPM"].to_numpy(float)
    dias = g["DIA_DESDE_INICIO"].to_numpy()
    fechas = g["FECHA"].to_numpy()
    tanque = g["ID_TANQUE"].to_numpy()
    evmat = {e: g[e].to_numpy(float) for e in EVENTOS}
    doy0 = pd.Timestamp(fechas[0]).dayofyear
    photo_start = daylength(doy0)
    for i, d in enumerate(dias):
        if d < MIN_DIA or d >= D: continue
        hph, hppm = ph[:i+1], ppm[:i+1]
        doy = pd.Timestamp(fechas[i]).dayofyear
        photo = daylength(doy)
        w3, w14 = slice(max(0,i-2),i+1), slice(max(0,i-13),i+1)
        w7 = slice(max(0,i-6),i+1)
        # aceleración PPM: slope reciente (3d) - slope previo (3d)
        prev3 = ppm[max(0,i-5):max(0,i-2)+1] if i>=3 else np.array([np.nan])
        accel = slope(ppm[w3]) - slope(prev3) if i>=3 else np.nan
        delta3 = hppm[-1] - np.nanmean(ppm[max(0,i-3):i]) if i>=1 else 0.0
        tk = tanque[:i+1]
        first_tk = np.where(~np.isnan(tanque))[0]
        dias_pretanque = int(first_tk[0]) if len(first_tk) else (i+1)
        n_ev_total = sum(np.nansum(evmat[e][:i+1]) for e in EVENTOS)
        n_ev_7 = sum(np.nansum(evmat[e][w7]) for e in EVENTOS)
        rows.append({
            "ID_COSECHA": cos, "ID_TANDA": tanda, "dia_actual": d,
            # FOTO
            "foto_actual": photo, "foto_inicio": photo_start,
            "foto_tend": photo - photo_start,
            # VENT
            "ph_slope3": slope(hph[w3] if False else ph[w3]), "ph_slope14": slope(ph[w14]),
            "ppm_slope3": slope(ppm[w3]), "ppm_slope14": slope(ppm[w14]),
            "ph_std7": np.nanstd(ph[w7]), "ph_std14": np.nanstd(ph[w14]),
            "ppm_std7": np.nanstd(ppm[w7]), "ppm_std14": np.nanstd(ppm[w14]),
            "ppm_accel": accel, "ppm_delta3": delta3,
            # RANGO
            "frac_ph_inrango": frac_inrango(hph, PH_LO, PH_HI),
            "frac_ppm_inrango": frac_inrango(hppm, PPM_LO, PPM_HI),
            # TANQUE
            "n_tanques_distintos": int(len(np.unique(tk[~np.isnan(tk)]))),
            "frac_dias_contanque": float(np.mean(~np.isnan(tk))),
            "dias_pretanque": dias_pretanque,
            # EVENT
            "tasa_eventos": n_ev_total/(i+1),
            "eventos_ultimos7": n_ev_7,
        })

NEW = pd.DataFrame(rows)
FAM = {
    "FOTO":   ["foto_actual","foto_inicio","foto_tend"],
    "VENT":   ["ph_slope3","ph_slope14","ppm_slope3","ppm_slope14","ph_std7","ph_std14",
               "ppm_std7","ppm_std14","ppm_accel","ppm_delta3"],
    "RANGO":  ["frac_ph_inrango","frac_ppm_inrango"],
    "TANQUE": ["n_tanques_distintos","frac_dias_contanque","dias_pretanque"],
    "EVENT":  ["tasa_eventos","eventos_ultimos7"],
}

# --- merge con el set del campeón ---
L = pd.read_parquet("landmarks_full.parquet")
raw_starts = df.groupby("ID_COSECHA")["FECHA"].min().rename("start")
L = L.merge(NEW, on=["ID_COSECHA","ID_TANDA","dia_actual"], how="left")
L = L.merge(raw_starts, on="ID_COSECHA", how="left")
matched = L[FAM["FOTO"][0]].notna().mean()
print(f"Landmarks: {len(L)} | match features nuevas: {matched*100:.1f}%")

DEFIN = [c for c in L.columns if c not in ["ID_COSECHA","ID_TANDA","D","start"]
         and c not in sum(FAM.values(), [])]
y = L["D"].to_numpy(float); groups = L["ID_COSECHA"].to_numpy()

LGB = dict(objective="regression_l1", n_estimators=800, learning_rate=0.03, num_leaves=63,
           max_depth=-1, min_child_samples=20, subsample=0.7, subsample_freq=1,
           colsample_bytree=0.6, reg_lambda=5, random_state=RNG, n_jobs=-1, verbose=-1)

def eval_cv(feats):
    oof = np.zeros(len(L))
    for tr, te in GroupKFold(5).split(L, y, groups):
        oof[te] = LGBMRegressor(**LGB).fit(L.iloc[tr][feats], y[tr]).predict(L.iloc[te][feats])
    return (mean_absolute_error(y, oof), median_absolute_error(y, oof),
            np.sqrt(mean_squared_error(y, oof)))

def eval_temporal(feats):
    tr = L["start"] < CUT; te = L["start"] >= CUT
    m = LGBMRegressor(**LGB).fit(L[tr][feats], y[tr])
    p = m.predict(L[te][feats]); yt = y[te]
    return (mean_absolute_error(yt, p), median_absolute_error(yt, p),
            np.sqrt(mean_squared_error(yt, p)))

SETS = {"DEFINITIVO (base+clima+historia)": DEFIN}
for fam, cols in FAM.items():
    SETS[f"+ {fam}"] = DEFIN + cols
SETS["+ TODAS las nuevas"] = DEFIN + sum(FAM.values(), [])

print("\n" + "="*82)
print("ABLACIÓN — aporte de las features nuevas (LightGBM tuneado)")
print("="*82)
print(f"{'Feature set':38s} | {'CV-MAE':>7} {'MedAE':>6} {'RMSE':>6} | {'Tmp-MAE':>7} {'RMSE':>6}")
base_cv = None
for name, feats in SETS.items():
    cv = eval_cv(feats); tmp = eval_temporal(feats)
    if base_cv is None: base_cv = cv[0]
    dcv = (cv[0]-base_cv)/base_cv*100
    flag = "" if name.startswith("DEFIN") else f"  ({dcv:+.1f}%)"
    print(f"{name:38s} | {cv[0]:7.3f} {cv[1]:6.2f} {cv[2]:6.2f} | {tmp[0]:7.3f} {tmp[2]:6.2f}{flag}")

print("\n(CV = GroupKFold por cosecha; Tmp = test temporal último año. % vs DEFINITIVO en CV.)")
