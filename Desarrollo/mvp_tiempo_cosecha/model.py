"""MVP: predicción del tiempo de cosecha, re-estimable día a día.

Enfoque "landmark": para cada tanda completa y cada día observado d, se arma
un ejemplo con features de la trayectoria hasta d, y el target es la DURACIÓN
TOTAL de la tanda (D). Predecir D es equivalente a predecir "días restantes"
(= D - d), porque d se conoce con exactitud.

Se comparan tres estrategias, todas validadas con GroupKFold agrupando por
ID_COSECHA (la unidad realmente independiente: 89 cosechas), para no mezclar
días ni tandas de una misma cosecha entre train y test:

  1. DUMMY        -> D_hat = duración media del train (ignora todo).
                     Equivale al modelo naif "resto = promedio - día actual".
  2. SOLO-ESTATICAS -> época del año + cultivo + día transcurrido (sin sensores).
  3. COMPLETO     -> estáticas + resumen de la trayectoria (pH, PPM, eventos).

La pregunta de viabilidad: ¿el MODELO le gana al DUMMY? ¿y las features de
sensores (COMPLETO) le ganan a solo-estáticas?
"""
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore", category=RuntimeWarning)  # nanmean de slices vacíos
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG = 42
CSV = "../dataset/dataset_albahaca_1.csv"
MIN_DIA = 3          # primer día en que se emite predicción (necesita algo de historia)
EVENTOS = ["TRASPASO_TANQUE", "AGREGADO_SALES", "CAMBIO_TANQUE", "CAMBIO_80_TANQUE",
           "BOVEL", "ESTEN_80", "G_A_F1"]

# ----------------------------------------------------------------------------
# 1. Cargar y quedarse con tandas COMPLETAS (con COSECHA_FIN observado)
# ----------------------------------------------------------------------------
df = pd.read_csv(CSV, parse_dates=["FECHA"])
# Normalizar PPM a una sola escala: mS/cm (hoja 2) -> ppm con factor 500 estándar.
# SUPUESTO a confirmar con la empresa (podría ser 640 según el medidor).
df["PPM"] = np.where(df["UNIDAD"].eq("mS/cm"), df["PPM"] * 500, df["PPM"])
# Sanear pH imposible (ej. 1620 en T59/FC534): fuera de [0,14] -> faltante.
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
print(f"Tandas completas: {len(completas)} | cosechas independientes: "
      f"{completas.index.get_level_values(0).nunique()}")

cult_cod = {c: i for i, c in enumerate(sorted(df["CULTIVO"].dropna().unique()))}
frac_cod = {f: i for i, f in enumerate(sorted(df["FRACCION"].astype(str).unique()))}

# ----------------------------------------------------------------------------
# 2. Construir el dataset de landmarks (un ejemplo por tanda x día observado)
# ----------------------------------------------------------------------------
def slope(y):
    y = np.asarray(y, float)
    m = ~np.isnan(y)
    if m.sum() < 2:
        return np.nan
    x = np.arange(len(y))[m]
    return np.polyfit(x, y[m], 1)[0]

def dias_desde(ev_arr, dias_arr, i):
    """Días transcurridos desde la última ocurrencia del evento hasta el día i
    (inclusive). NaN si el evento todavía no ocurrió."""
    ocurr = np.where(np.nan_to_num(ev_arr[:i + 1]) == 1)[0]
    if len(ocurr) == 0:
        return np.nan
    return float(dias_arr[i] - dias_arr[ocurr[-1]])

rows = []
for (cos, tanda), g in df.groupby(["ID_COSECHA", "ID_TANDA"]):
    if (cos, tanda) not in completas.index:
        continue
    D = completas.loc[(cos, tanda), "duracion"]
    mes = completas.loc[(cos, tanda), "mes_inicio"]
    cult = completas.loc[(cos, tanda), "cultivo"]
    frac = completas.loc[(cos, tanda), "fraccion"]
    g = g.sort_values("DIA_DESDE_INICIO")
    ph = g["PH"].to_numpy(float)
    ppm = g["PPM"].to_numpy(float)
    dias = g["DIA_DESDE_INICIO"].to_numpy()
    ev = {e: g[e].to_numpy() for e in EVENTOS}
    tanque = g["ID_TANQUE"].to_numpy()  # NaN si pre-tanque

    for i, d in enumerate(dias):
        if d < MIN_DIA or d >= D:   # no emitir el día del evento en sí
            continue
        h_ph, h_ppm = ph[:i + 1], ppm[:i + 1]
        w = slice(max(0, i - 6), i + 1)  # ventana últimos 7 días
        feat = {
            "ID_COSECHA": cos, "ID_TANDA": tanda, "D": D, "dia_actual": d,
            # estáticas (conocidas desde el inicio)
            "mes_inicio": mes,
            "mes_sin": np.sin(2 * np.pi * mes / 12),
            "mes_cos": np.cos(2 * np.pi * mes / 12),
            "cultivo_cod": cult_cod.get(cult, -1),
            "fraccion_cod": frac_cod.get(str(frac), -1),
            # trayectoria pH
            "ph_last": h_ph[-1], "ph_mean": np.nanmean(h_ph),
            "ph_std": np.nanstd(h_ph), "ph_min": np.nanmin(h_ph),
            "ph_max": np.nanmax(h_ph), "ph_slope": slope(h_ph),
            "ph_mean7": np.nanmean(ph[w]),
            # trayectoria PPM
            "ppm_last": h_ppm[-1], "ppm_mean": np.nanmean(h_ppm),
            "ppm_std": np.nanstd(h_ppm), "ppm_min": np.nanmin(h_ppm),
            "ppm_max": np.nanmax(h_ppm), "ppm_slope": slope(h_ppm),
            "ppm_mean7": np.nanmean(ppm[w]),
            # estado de tanque
            "tiene_tanque": int(not np.isnan(tanque[i])),
            "dias_en_tanque": int(np.sum(~np.isnan(tanque[:i + 1]))),
            # eventos acumulados hasta d (conteo)
            **{f"n_{e}": int(np.nansum(ev[e][:i + 1])) for e in EVENTOS},
            # recencia: días desde la última vez que ocurrió el evento (NaN si nunca)
            **{f"dias_desde_{e}": dias_desde(ev[e], dias, i) for e in EVENTOS},
        }
        rows.append(feat)

L = pd.DataFrame(rows)
print(f"Landmarks (ejemplos tanda x día): {len(L)}")
L.to_parquet("landmarks.parquet")

STATIC = ["mes_inicio", "mes_sin", "mes_cos", "cultivo_cod", "fraccion_cod", "dia_actual"]
TRAJ = [c for c in L.columns if c not in STATIC + ["ID_COSECHA", "ID_TANDA", "D"]]
FEATS_FULL = STATIC + TRAJ
y = L["D"].to_numpy(float)
groups = L["ID_COSECHA"].to_numpy()

# ----------------------------------------------------------------------------
# 3. Validación cruzada agrupada por cosecha
# ----------------------------------------------------------------------------
def nuevo_modelo():
    return HistGradientBoostingRegressor(
        max_depth=4, max_iter=300, learning_rate=0.05,
        min_samples_leaf=40, l2_regularization=1.0, random_state=RNG)

gkf = GroupKFold(n_splits=5)
oof = {"dummy": np.zeros(len(L)), "estaticas": np.zeros(len(L)),
       "completo": np.zeros(len(L))}
fold_mae = {"dummy": [], "estaticas": [], "completo": []}

for tr, te in gkf.split(L, y, groups):
    dur_media = y[tr].mean()
    oof["dummy"][te] = dur_media

    m_est = nuevo_modelo().fit(L.iloc[tr][STATIC], y[tr])
    oof["estaticas"][te] = m_est.predict(L.iloc[te][STATIC])

    m_full = nuevo_modelo().fit(L.iloc[tr][FEATS_FULL], y[tr])
    oof["completo"][te] = m_full.predict(L.iloc[te][FEATS_FULL])

    fold_mae["dummy"].append(mean_absolute_error(y[te], oof["dummy"][te]))
    fold_mae["estaticas"].append(mean_absolute_error(y[te], oof["estaticas"][te]))
    fold_mae["completo"].append(mean_absolute_error(y[te], oof["completo"][te]))

print("\n================  RESULTADOS (MAE en días, out-of-fold)  ================")
print("MAE sobre 'duración total' == MAE sobre 'días restantes' (d es exacto).\n")
res = {}
for k in ["dummy", "estaticas", "completo"]:
    mae = mean_absolute_error(y, oof[k])
    medae = np.median(np.abs(y - oof[k]))
    res[k] = mae
    fm = np.array(fold_mae[k])
    print(f"  {k:12s}  MAE = {mae:5.2f} d   |  MedAE = {medae:5.2f} d"
          f"   |  por fold: {fm.mean():.2f} ± {fm.std():.2f}  (rango {fm.min():.2f}-{fm.max():.2f})")

mej_est = 100 * (res["dummy"] - res["estaticas"]) / res["dummy"]
mej_full = 100 * (res["dummy"] - res["completo"]) / res["dummy"]
mej_traj = 100 * (res["estaticas"] - res["completo"]) / res["estaticas"]
print(f"\n  Mejora estáticas vs dummy : {mej_est:5.1f}%")
print(f"  Mejora completo  vs dummy : {mej_full:5.1f}%")
print(f"  Aporte trayectoria (completo vs estáticas): {mej_traj:5.1f}%")

# ----------------------------------------------------------------------------
# 4. MAE por etapa del ciclo (¿dónde aportan las features?)
# ----------------------------------------------------------------------------
print("\n===============  MAE por día transcurrido (etapa del ciclo)  ==============")
L = L.assign(**{f"pred_{k}": oof[k] for k in oof})
buckets = [(3, 10), (11, 20), (21, 30), (31, 45), (46, 100)]
print(f"  {'etapa (días)':14s} {'n':>6s} {'dummy':>8s} {'estát.':>8s} {'completo':>9s}")
rows_plot = []
for lo, hi in buckets:
    m = (L["dia_actual"] >= lo) & (L["dia_actual"] <= hi)
    if m.sum() == 0:
        continue
    md = mean_absolute_error(y[m.values], oof["dummy"][m.values])
    me = mean_absolute_error(y[m.values], oof["estaticas"][m.values])
    mf = mean_absolute_error(y[m.values], oof["completo"][m.values])
    print(f"  {f'{lo}-{hi}':14s} {int(m.sum()):6d} {md:8.2f} {me:8.2f} {mf:9.2f}")
    rows_plot.append(((lo + hi) / 2, md, me, mf))

# ----------------------------------------------------------------------------
# 5. Importancia por permutación (modelo completo, refit en todo)
# ----------------------------------------------------------------------------
from sklearn.inspection import permutation_importance
m_all = nuevo_modelo().fit(L[FEATS_FULL], y)
pi = permutation_importance(m_all, L[FEATS_FULL], y, n_repeats=5,
                            random_state=RNG, scoring="neg_mean_absolute_error")
imp = pd.Series(pi.importances_mean, index=FEATS_FULL).sort_values(ascending=False)
imp_std = pd.Series(pi.importances_std, index=FEATS_FULL)
print("\n=========  Feature importance COMPLETO (permutation, todas las features)  =========")
print("  (aumento del MAE en días al permutar la feature; mayor = más importante)\n")
for name, val in imp.items():
    print(f"  {name:20s} {val:7.3f}  ± {imp_std[name]:.3f}")

# Barplot del FI
plt.figure(figsize=(8, 8))
order = imp.sort_values()
colors = ["#2f7de8" if f in STATIC else "#e8952f" for f in order.index]
plt.barh(order.index, order.values, xerr=imp_std[order.index], color=colors)
plt.xlabel("Importancia (aumento del MAE al permutar, días)")
plt.title("Feature importance — modelo completo (azul=estática, naranja=trayectoria)")
plt.grid(alpha=0.3, axis="x")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=130)
print("\nGráfico guardado: feature_importance.png")

# ----------------------------------------------------------------------------
# 6. Gráfico: MAE por etapa, modelo vs dummy
# ----------------------------------------------------------------------------
arr = np.array(rows_plot)
plt.figure(figsize=(8, 5))
plt.plot(arr[:, 0], arr[:, 1], "o--", label="Dummy (duración = promedio)", color="#999")
plt.plot(arr[:, 0], arr[:, 2], "s-", label="Solo estáticas (época+cultivo)", color="#e8952f")
plt.plot(arr[:, 0], arr[:, 3], "^-", label="Modelo completo (+ trayectoria)", color="#2f7de8")
plt.xlabel("Día del ciclo en que se predice")
plt.ylabel("MAE de la duración total (días)")
plt.title("Error de predicción del tiempo de cosecha por etapa del ciclo")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("mae_por_etapa.png", dpi=130)
print("\nGráfico guardado: mae_por_etapa.png")
