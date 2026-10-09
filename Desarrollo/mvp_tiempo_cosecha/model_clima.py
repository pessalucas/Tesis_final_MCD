"""EXP-D1 · MVP con enrichment climático.

Extiende el MVP landmark (ver model.py) agregando features de CLIMA, y mide si bajan
el MAE respecto del modelo sin clima. Misma tecnología: HistGradientBoostingRegressor,
GroupKFold por ID_COSECHA, target = duración total D (≡ días restantes).

Fuentes de clima (../dataset_enrichment/), con diseño anti-leakage:
  - era5_real       -> clima PASADO del ciclo (días 1..d) + ventana previa -60d. Sin fuga.
  - climatologia    -> clima ESPERADO del futuro (normales por día del año).
  - pronostico_lead -> PRONÓSTICO a 7 días emitido el día d (con error real).

Comparación: dummy / sin_clima (base MVP) / con_clima. Aísla el aporte del clima.
"""
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore", category=RuntimeWarning)
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
from sklearn.inspection import permutation_importance
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG = 42
MIN_DIA = 3
DS = "../dataset/dataset_albahaca_1.csv"
ENR = "../dataset_enrichment"
EVENTOS = ["TRASPASO_TANQUE", "AGREGADO_SALES", "CAMBIO_TANQUE", "CAMBIO_80_TANQUE",
           "BOVEL", "ESTEN_80", "G_A_F1"]

# --------------------------------------------------------------------------- #
# Carga y limpieza (igual que model.py)
# --------------------------------------------------------------------------- #
df = pd.read_csv(DS, parse_dates=["FECHA"])
df["PPM"] = np.where(df["UNIDAD"].eq("mS/cm"), df["PPM"] * 500, df["PPM"])
df.loc[(df["PH"] < 0) | (df["PH"] > 14), "PH"] = np.nan
df = df.sort_values(["ID_COSECHA", "ID_TANDA", "DIA_DESDE_INICIO"]).reset_index(drop=True)

def info_tanda(sub):
    fin = sub[sub["COSECHA_FIN"] == 1]
    dur = fin["DIA_DESDE_INICIO"].min() if len(fin) else np.nan
    return pd.Series({"duracion": dur, "cultivo": sub["CULTIVO"].iloc[0],
                      "fraccion": str(sub["FRACCION"].iloc[0]),
                      "mes_inicio": sub["FECHA"].min().month})

meta = df.groupby(["ID_COSECHA", "ID_TANDA"]).apply(info_tanda, include_groups=False)
completas = meta[meta["duracion"].notna()].copy()
cult_cod = {c: i for i, c in enumerate(sorted(df["CULTIVO"].dropna().unique()))}
frac_cod = {f: i for i, f in enumerate(sorted(df["FRACCION"].astype(str).unique()))}

# --------------------------------------------------------------------------- #
# Enrichment climático: series indexadas por fecha
# --------------------------------------------------------------------------- #
era5 = pd.read_csv(f"{ENR}/era5_real/clima_era5_diario.csv", parse_dates=["FECHA"]).set_index("FECHA")
clim = pd.read_csv(f"{ENR}/climatologia/climatologia_doy.csv").set_index("doy")
pron = pd.read_csv(f"{ENR}/pronostico_lead/pronostico_lead_por_emision.csv", parse_dates=["FECHA"]).set_index("FECHA")

# clima esperado por FECHA (climatología mapeada al día del año) -> ventanas futuras y anomalías
rango = pd.date_range(df["FECHA"].min() - pd.Timedelta(days=70),
                      df["FECHA"].max() + pd.Timedelta(days=90), freq="D")
exp = pd.DataFrame(index=rango)
for col, src in [("dli", "dli_mol_m2"), ("temp", "temperature_2m_mean"), ("gdd", "gdd_base10")]:
    exp[col] = clim[src].reindex(exp.index.dayofyear).values
exp_dli_cs, exp_temp_cs, exp_gdd_cs = exp["dli"].cumsum(), exp["temp"].cumsum(), exp["gdd"].cumsum()

def fut_mean(cs, F, n):
    try: return (cs.loc[F + pd.Timedelta(days=n)] - cs.loc[F]) / n
    except KeyError: return np.nan

def fut_sum(cs, F, n):
    try: return cs.loc[F + pd.Timedelta(days=n)] - cs.loc[F]
    except KeyError: return np.nan

# series era5 alineables por fecha
e_temp, e_dli, e_gdd = era5["temperature_2m_mean"], era5["dli_mol_m2"], era5["gdd_base10"]
e_prec, e_cloud = era5["precipitation_sum"], era5["cloud_cover_mean"]
e_rh, e_vpd = era5["relative_humidity_2m_mean"], era5["vapour_pressure_deficit_max"]
e_et0, e_sun, e_ampl = era5["et0_fao_evapotranspiration"], era5["sunshine_h"], era5["temp_amplitud"]
# climatología por fecha (para anomalías: real - típico)
cd_temp = clim["temperature_2m_mean"].reindex(rango.dayofyear).set_axis(rango)
cd_dli = clim["dli_mol_m2"].reindex(rango.dayofyear).set_axis(rango)
cd_gdd = clim["gdd_base10"].reindex(rango.dayofyear).set_axis(rango)

# --------------------------------------------------------------------------- #
# Construcción de landmarks con features base + clima
# --------------------------------------------------------------------------- #
def slope(y):
    y = np.asarray(y, float); m = ~np.isnan(y)
    if m.sum() < 2: return np.nan
    return np.polyfit(np.arange(len(y))[m], y[m], 1)[0]

def dias_desde(ev_arr, dias_arr, i):
    oc = np.where(np.nan_to_num(ev_arr[:i + 1]) == 1)[0]
    return np.nan if len(oc) == 0 else float(dias_arr[i] - dias_arr[oc[-1]])

rows = []
for (cos, tanda), g in df.groupby(["ID_COSECHA", "ID_TANDA"]):
    if (cos, tanda) not in completas.index:
        continue
    D = completas.loc[(cos, tanda), "duracion"]
    mes = completas.loc[(cos, tanda), "mes_inicio"]
    cult = completas.loc[(cos, tanda), "cultivo"]
    frac = completas.loc[(cos, tanda), "fraccion"]
    g = g.sort_values("DIA_DESDE_INICIO")
    ph, ppm = g["PH"].to_numpy(float), g["PPM"].to_numpy(float)
    dias = g["DIA_DESDE_INICIO"].to_numpy()
    fechas = g["FECHA"].to_numpy()
    F0 = pd.Timestamp(fechas[0])
    ev = {e: g[e].to_numpy() for e in EVENTOS}
    tanque = g["ID_TANQUE"].to_numpy()

    # clima real alineado a los días de la tanda
    fidx = pd.DatetimeIndex(fechas)
    c_temp = e_temp.reindex(fidx).to_numpy(); c_dli = e_dli.reindex(fidx).to_numpy()
    c_gdd = e_gdd.reindex(fidx).to_numpy(); c_prec = e_prec.reindex(fidx).to_numpy()
    c_cloud = e_cloud.reindex(fidx).to_numpy(); c_rh = e_rh.reindex(fidx).to_numpy()
    c_vpd = e_vpd.reindex(fidx).to_numpy(); c_et0 = e_et0.reindex(fidx).to_numpy()
    c_sun = e_sun.reindex(fidx).to_numpy(); c_ampl = e_ampl.reindex(fidx).to_numpy()
    # climatología alineada a los mismos días (para anomalías)
    k_temp = cd_temp.reindex(fidx).to_numpy(); k_dli = cd_dli.reindex(fidx).to_numpy()
    k_gdd = cd_gdd.reindex(fidx).to_numpy()
    # ventana previa -60d (antecedentes, una vez por tanda)
    prev = pd.date_range(F0 - pd.Timedelta(days=60), F0 - pd.Timedelta(days=1), freq="D")
    prev60_temp = e_temp.reindex(prev).mean()
    prev60_dli = e_dli.reindex(prev).mean()
    prev60_dli_anom = prev60_dli - cd_dli.reindex(prev).mean()

    for i, d in enumerate(dias):
        if d < MIN_DIA or d >= D:
            continue
        h_ph, h_ppm = ph[:i + 1], ppm[:i + 1]
        w = slice(max(0, i - 6), i + 1)
        F = pd.Timestamp(fechas[i])
        feat = {
            "ID_COSECHA": cos, "ID_TANDA": tanda, "D": D, "dia_actual": d,
            "mes_inicio": mes, "mes_sin": np.sin(2*np.pi*mes/12), "mes_cos": np.cos(2*np.pi*mes/12),
            "cultivo_cod": cult_cod.get(cult, -1), "fraccion_cod": frac_cod.get(str(frac), -1),
            # --- trayectoria pH/PPM (base MVP) ---
            "ph_last": h_ph[-1], "ph_mean": np.nanmean(h_ph), "ph_std": np.nanstd(h_ph),
            "ph_min": np.nanmin(h_ph), "ph_max": np.nanmax(h_ph), "ph_slope": slope(h_ph),
            "ph_mean7": np.nanmean(ph[w]),
            "ppm_last": h_ppm[-1], "ppm_mean": np.nanmean(h_ppm), "ppm_std": np.nanstd(h_ppm),
            "ppm_min": np.nanmin(h_ppm), "ppm_max": np.nanmax(h_ppm), "ppm_slope": slope(h_ppm),
            "ppm_mean7": np.nanmean(ppm[w]),
            "tiene_tanque": int(not np.isnan(tanque[i])),
            "dias_en_tanque": int(np.sum(~np.isnan(tanque[:i + 1]))),
            **{f"n_{e}": int(np.nansum(ev[e][:i + 1])) for e in EVENTOS},
            **{f"dias_desde_{e}": dias_desde(ev[e], dias, i) for e in EVENTOS},
            # --- CLIMA: pasado del ciclo (real, sin fuga) ---
            "cl_temp_mean_ciclo": np.nanmean(c_temp[:i + 1]),
            "cl_temp_min_ciclo": np.nanmin(c_temp[:i + 1]),
            "cl_temp_max_ciclo": np.nanmax(c_temp[:i + 1]),
            "cl_temp_std_ciclo": np.nanstd(c_temp[:i + 1]),
            "cl_temp_slope_ciclo": slope(c_temp[:i + 1]),
            "cl_dli_mean_ciclo": np.nanmean(c_dli[:i + 1]),
            "cl_dli_min_ciclo": np.nanmin(c_dli[:i + 1]),
            "cl_dli_std_ciclo": np.nanstd(c_dli[:i + 1]),
            "cl_dli_slope_ciclo": slope(c_dli[:i + 1]),
            "cl_dli_acum_ciclo": np.nansum(c_dli[:i + 1]),
            "cl_gdd_acum_ciclo": np.nansum(c_gdd[:i + 1]),
            "cl_prec_acum_ciclo": np.nansum(c_prec[:i + 1]),
            "cl_cloud_mean_ciclo": np.nanmean(c_cloud[:i + 1]),
            "cl_rh_mean_ciclo": np.nanmean(c_rh[:i + 1]),
            "cl_vpd_mean_ciclo": np.nanmean(c_vpd[:i + 1]),
            "cl_et0_acum_ciclo": np.nansum(c_et0[:i + 1]),
            "cl_sun_mean_ciclo": np.nanmean(c_sun[:i + 1]),
            "cl_ampl_mean_ciclo": np.nanmean(c_ampl[:i + 1]),
            "cl_ndias_frios": int(np.nansum(c_temp[:i + 1] < 10)),
            "cl_ndias_pocaluz": int(np.nansum(c_dli[:i + 1] < 20)),
            # ventanas recientes (ritmo actual)
            "cl_dli_last3": np.nanmean(c_dli[max(0, i - 2):i + 1]),
            "cl_dli_last7": np.nanmean(c_dli[w]),
            "cl_dli_last14": np.nanmean(c_dli[max(0, i - 13):i + 1]),
            "cl_temp_last7": np.nanmean(c_temp[w]),
            "cl_temp_last14": np.nanmean(c_temp[max(0, i - 13):i + 1]),
            # --- CLIMA: ANOMALÍAS (real - típico de la época) ---
            "cl_anom_temp_ciclo": np.nanmean(c_temp[:i + 1]) - np.nanmean(k_temp[:i + 1]),
            "cl_anom_dli_ciclo": np.nanmean(c_dli[:i + 1]) - np.nanmean(k_dli[:i + 1]),
            "cl_anom_temp_last7": np.nanmean(c_temp[w]) - np.nanmean(k_temp[w]),
            "cl_anom_dli_last7": np.nanmean(c_dli[w]) - np.nanmean(k_dli[w]),
            "cl_gdd_acum_vs_clim": np.nansum(c_gdd[:i + 1]) - np.nansum(k_gdd[:i + 1]),
            "cl_dli_acum_vs_clim": np.nansum(c_dli[:i + 1]) - np.nansum(k_dli[:i + 1]),
            # --- CLIMA: antecedentes -60d ---
            "cl_prev60_temp": prev60_temp,
            "cl_prev60_dli": prev60_dli,
            "cl_prev60_dli_anom": prev60_dli_anom,
            # --- CLIMA: futuro esperado (climatología por día del año) ---
            "cl_fut7_dli_clim": fut_mean(exp_dli_cs, F, 7),
            "cl_fut14_dli_clim": fut_mean(exp_dli_cs, F, 14),
            "cl_fut30_dli_clim": fut_mean(exp_dli_cs, F, 30),
            "cl_fut30_temp_clim": fut_mean(exp_temp_cs, F, 30),
            "cl_fut30_gdd_clim": fut_sum(exp_gdd_cs, F, 30),
            "cl_dli_tendencia": fut_mean(exp_dli_cs, F, 14) - np.nanmean(c_dli[w]),  # ¿mejora la luz?
            # --- CLIMA: pronóstico a 7 días (con error real, emitido el día d) ---
            "cl_fc_temp_d1": pron["temperature_2m__d1"].get(F, np.nan),
            "cl_fc_temp_d7": pron["temperature_2m__d7"].get(F, np.nan),
            "cl_fc_dli_d1": pron["dli__d1"].get(F, np.nan),
            "cl_fc_dli_d7": pron["dli__d7"].get(F, np.nan),
            "cl_fc_temp_next7": pron["temperature_2m__next7"].get(F, np.nan),
            "cl_fc_dli_next7": pron["dli__next7"].get(F, np.nan),
            "cl_fc_prec_next7": pron["precipitation__next7"].get(F, np.nan),
            "cl_fc_cloud_next7": pron["cloud_cover__next7"].get(F, np.nan),
            # anomalía del pronóstico: ¿la próxima semana viene más luminosa que lo normal?
            "cl_fc_dli_next7_anom": pron["dli__next7"].get(F, np.nan) - fut_mean(exp_dli_cs, F, 7),
        }
        rows.append(feat)

L = pd.DataFrame(rows)
print(f"Landmarks: {len(L)} | cosechas: {L['ID_COSECHA'].nunique()}")
L.to_parquet("landmarks_clima.parquet")  # para reutilizar (tuning, etc.)

BASE_STATIC = ["mes_inicio", "mes_sin", "mes_cos", "cultivo_cod", "fraccion_cod", "dia_actual"]
CLIMA = [c for c in L.columns if c.startswith("cl_")]
NO_FEAT = ["ID_COSECHA", "ID_TANDA", "D"]
TRAJ = [c for c in L.columns if c not in BASE_STATIC + CLIMA + NO_FEAT]
# set climático CURADO: las mejores + anomalías conceptualmente fuertes, sin relleno
CLIMA_SEL = ["cl_fc_dli_next7", "cl_prev60_temp", "cl_prev60_dli", "cl_prev60_dli_anom",
             "cl_dli_min_ciclo", "cl_fut7_dli_clim", "cl_fut14_dli_clim",
             "cl_dli_mean_ciclo", "cl_gdd_acum_ciclo", "cl_anom_dli_ciclo",
             "cl_gdd_acum_vs_clim", "cl_fc_dli_next7_anom", "cl_cloud_mean_ciclo"]
FEATS_SIN = BASE_STATIC + TRAJ                    # base MVP (sin clima)
FEATS_CON = BASE_STATIC + TRAJ + CLIMA            # con clima (todas, 49)
FEATS_SEL = BASE_STATIC + TRAJ + CLIMA_SEL        # con clima curado (13)
print(f"Features sin clima: {len(FEATS_SIN)} | con clima (todas): {len(FEATS_CON)} "
      f"(+{len(CLIMA)}) | con clima curado: {len(FEATS_SEL)} (+{len(CLIMA_SEL)})")

y = L["D"].to_numpy(float)
groups = L["ID_COSECHA"].to_numpy()

def nuevo():
    return HistGradientBoostingRegressor(max_depth=4, max_iter=300, learning_rate=0.05,
                                         min_samples_leaf=40, l2_regularization=1.0, random_state=RNG)

gkf = GroupKFold(n_splits=5)
FEATSETS = {"sin_clima": FEATS_SIN, "con_clima": FEATS_CON, "clima_curado": FEATS_SEL}
oof = {k: np.zeros(len(L)) for k in ["dummy"] + list(FEATSETS)}
fold_mae = {k: [] for k in oof}
for tr, te in gkf.split(L, y, groups):
    oof["dummy"][te] = y[tr].mean()
    for k, feats in FEATSETS.items():
        oof[k][te] = nuevo().fit(L.iloc[tr][feats], y[tr]).predict(L.iloc[te][feats])
    for k in oof:
        fold_mae[k].append(mean_absolute_error(y[te], oof[k][te]))

print("\n================  RESULTADOS (MAE en días, OOF)  ================")
res = {}
for k in ["dummy", "sin_clima", "con_clima", "clima_curado"]:
    mae = mean_absolute_error(y, oof[k]); res[k] = mae
    fm = np.array(fold_mae[k])
    print(f"  {k:13s} MAE={mae:5.2f} d | MedAE={np.median(np.abs(y-oof[k])):4.2f} | "
          f"folds {fm.mean():.2f}±{fm.std():.2f} ({fm.min():.2f}-{fm.max():.2f})")
print(f"\n  Aporte clima (todas)  vs sin: {100*(res['sin_clima']-res['con_clima'])/res['sin_clima']:+.1f}%")
print(f"  Aporte clima (curado) vs sin: {100*(res['sin_clima']-res['clima_curado'])/res['sin_clima']:+.1f}%")

# MAE por etapa
print("\n===============  MAE por día del ciclo  ===============")
print(f"  {'etapa':10s} {'n':>6s} {'dummy':>7s} {'sin_cl':>7s} {'curado':>7s}")
plot_rows = []
for lo, hi in [(3,10),(11,20),(21,30),(31,45),(46,100)]:
    m = ((L["dia_actual"]>=lo)&(L["dia_actual"]<=hi)).values
    if m.sum()==0: continue
    md,ms,mc = (mean_absolute_error(y[m],oof[k][m]) for k in ["dummy","sin_clima","clima_curado"])
    print(f"  {f'{lo}-{hi}':10s} {int(m.sum()):6d} {md:7.2f} {ms:7.2f} {mc:7.2f}")
    plot_rows.append(((lo+hi)/2, md, ms, mc))

# Importancia de features (modelo curado)
m_all = nuevo().fit(L[FEATS_SEL], y)
pi = permutation_importance(m_all, L[FEATS_SEL], y, n_repeats=5, random_state=RNG,
                            scoring="neg_mean_absolute_error")
imp = pd.Series(pi.importances_mean, index=FEATS_SEL).sort_values(ascending=False)
print("\n=======  Top 20 features (permutation) — clima marcado con *  =======")
for name, val in imp.head(20).items():
    print(f"  {'*' if name in CLIMA else ' '} {name:24s} {val:6.3f}")
print(f"\n  Features climáticas en el top 20: {sum(1 for n in imp.head(20).index if n in CLIMA)}/20")

# Gráfico
arr = np.array(plot_rows)
plt.figure(figsize=(8,5))
plt.plot(arr[:,0], arr[:,1], "o--", color="#999", label="Dummy")
plt.plot(arr[:,0], arr[:,2], "s-", color="#e8952f", label="MVP sin clima")
plt.plot(arr[:,0], arr[:,3], "^-", color="#2f7de8", label="MVP + clima")
plt.xlabel("Día del ciclo en que se predice"); plt.ylabel("MAE duración total (días)")
plt.title("EXP-D1 · Aporte del enrichment climático por etapa")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig("mae_clima_por_etapa.png", dpi=130)
print("\nGráfico: mae_clima_por_etapa.png")
