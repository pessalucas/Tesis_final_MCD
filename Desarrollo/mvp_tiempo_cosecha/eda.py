"""EDA de tesis — distribución del target, consistencia temporal, TRAIN vs TEST,
y relación features<->target. Genera figuras PNG + un resumen `eda.md`.

Correr: uv run eda.py
"""
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

CSV = "../dataset/dataset_albahaca_1.csv"
CUT = pd.Timestamp("2025-09-26")   # test = último año
BLUE, ORANGE = "#2f7de8", "#e8952f"
out = []  # líneas para eda.md
def log(s=""): out.append(str(s)); print(s)

df = pd.read_csv(CSV, parse_dates=["FECHA"])
df["PPM"] = np.where(df["UNIDAD"].eq("mS/cm"), df["PPM"] * 500, df["PPM"])
df.loc[(df["PH"] < 0) | (df["PH"] > 14), "PH"] = np.nan

# ---------- nivel COSECHA/TANDA: duración (target) ----------
def dur_tanda(sub):
    fin = sub[sub["COSECHA_FIN"] == 1]
    return pd.Series({
        "duracion": fin["DIA_DESDE_INICIO"].min() if len(fin) else np.nan,
        "inicio": sub["FECHA"].min(), "mes": sub["FECHA"].min().month,
        "anio": sub["FECHA"].min().year, "n_mediciones": len(sub),
        "ph_mean": sub["PH"].mean(), "ppm_mean": sub["PPM"].mean(),
        "hoja": sub["HOJA"].iloc[0]})
t = df.groupby(["ID_COSECHA", "ID_TANDA"]).apply(dur_tanda, include_groups=False)
comp = t[t["duracion"].notna()].copy()
comp["split"] = np.where(comp["inicio"] < CUT, "train", "test")

log("# EDA — Predicción del tiempo de cosecha (albahaca)\n")
log(f"Tandas completas: {len(comp)} | cosechas: {comp.index.get_level_values(0).nunique()}")
log(f"Rango temporal: {comp['inicio'].min().date()} -> {comp['inicio'].max().date()}\n")

# === 1. DISTRIBUCIÓN DEL TARGET ===
log("## 1. Distribución del target (duración total, días)")
d = comp["duracion"]
log(f"- media {d.mean():.1f} | mediana {d.median():.1f} | sd {d.std():.1f} | "
    f"min {d.min():.0f} | max {d.max():.0f}")
log(f"- rango intercuartil: [{d.quantile(.25):.0f}, {d.quantile(.75):.0f}] | "
    f"asimetría (skew) {d.skew():.2f} | curtosis {d.kurtosis():.2f}")
log(f"- Interpretación: skew {d.skew():+.2f} => "
    f"{'cola derecha (algunas cosechas muy largas)' if d.skew()>0.3 else 'aprox. simétrica'}.\n")

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].hist(d, bins=30, color=BLUE, edgecolor="white")
ax[0].axvline(d.mean(), color="red", ls="--", label=f"media {d.mean():.0f}")
ax[0].axvline(d.median(), color="green", ls="--", label=f"mediana {d.median():.0f}")
ax[0].set_title("Distribución de la duración total"); ax[0].set_xlabel("días"); ax[0].legend()
comp.boxplot(column="duracion", by="mes", ax=ax[1])
ax[1].set_title("Duración por mes de inicio (estacionalidad)"); ax[1].set_xlabel("mes"); ax[1].set_ylabel("días")
plt.suptitle(""); plt.tight_layout(); plt.savefig("eda_target.png", dpi=130); plt.close()

# === 2. CONSISTENCIA A LO LARGO DEL TIEMPO ===
log("## 2. Consistencia a lo largo del tiempo")
por_anio = comp.groupby("anio")["duracion"].agg(["count", "median", "mean", "std"])
log("Duración por año de inicio:")
log(por_anio.to_string() + "\n")
dens = comp.groupby("anio")["n_mediciones"].median()
log("Mediana de mediciones por tanda (densidad de registro) por año:")
log(dens.to_string() + "\n")
# faltantes de pH/PPM por año (a nivel fila)
df["anio"] = df["FECHA"].dt.year
falt = df.groupby("anio").agg(ph_falt=("PH", lambda s: s.isna().mean()*100),
                              ppm_falt=("PPM", lambda s: s.isna().mean()*100))
log("Faltantes (%) por año:")
log(falt.round(1).to_string() + "\n")
# escala de PPM en el tiempo (hoja1 vs hoja2 / unidad)
log("Unidad de PPM por hoja (el shift de instrumentación):")
log(df.groupby("HOJA")["UNIDAD"].value_counts().to_string() + "\n")

fig, ax = plt.subplots(2, 2, figsize=(12, 8))
por_anio["median"].plot(kind="bar", ax=ax[0, 0], color=BLUE)
ax[0, 0].set_title("Duración mediana por año"); ax[0, 0].set_ylabel("días")
comp.groupby(comp["inicio"].dt.to_period("Q")).size().plot(kind="bar", ax=ax[0, 1], color=ORANGE)
ax[0, 1].set_title("Cosechas iniciadas por trimestre")
dens.plot(kind="bar", ax=ax[1, 0], color=BLUE)
ax[1, 0].set_title("Mediciones por tanda (mediana) por año")
falt.plot(kind="line", marker="o", ax=ax[1, 1])
ax[1, 1].set_title("% faltantes por año"); ax[1, 1].set_ylabel("%")
plt.tight_layout(); plt.savefig("eda_consistencia_temporal.png", dpi=130); plt.close()

# === 3. TRAIN vs TEST ===
log("## 3. Consistencia TRAIN vs TEST (último año)")
tr, te = comp[comp["split"] == "train"], comp[comp["split"] == "test"]
log(f"TRAIN: {len(tr)} tandas | TEST: {len(te)} tandas\n")
tab = pd.DataFrame({
    "TRAIN": [tr["duracion"].mean(), tr["duracion"].median(), tr["duracion"].std(),
              tr["ppm_mean"].mean(), tr["ph_mean"].mean()],
    "TEST":  [te["duracion"].mean(), te["duracion"].median(), te["duracion"].std(),
              te["ppm_mean"].mean(), te["ph_mean"].mean()]},
    index=["dur_media", "dur_mediana", "dur_sd", "ppm_medio", "ph_medio"])
log(tab.round(2).to_string() + "\n")
ks = stats.ks_2samp(tr["duracion"], te["duracion"])
log(f"Test KS de la duración (TRAIN vs TEST): D={ks.statistic:.3f}, p={ks.pvalue:.3f} "
    f"=> {'distribuciones DIFERENTES (hay shift)' if ks.pvalue<0.05 else 'sin evidencia de shift'}\n")

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].hist(tr["duracion"], bins=25, density=True, alpha=0.55, color=BLUE, label=f"train (n={len(tr)})")
ax[0].hist(te["duracion"], bins=25, density=True, alpha=0.55, color=ORANGE, label=f"test (n={len(te)})")
ax[0].set_title("Duración: TRAIN vs TEST"); ax[0].set_xlabel("días"); ax[0].legend()
ax[1].hist(tr["ppm_mean"].dropna(), bins=25, density=True, alpha=0.55, color=BLUE, label="train")
ax[1].hist(te["ppm_mean"].dropna(), bins=25, density=True, alpha=0.55, color=ORANGE, label="test")
ax[1].set_title("PPM medio por tanda: TRAIN vs TEST"); ax[1].set_xlabel("ppm"); ax[1].legend()
plt.tight_layout(); plt.savefig("eda_train_vs_test.png", dpi=130); plt.close()

# === 4. RELACIÓN FEATURES <-> TARGET (nivel landmark) ===
log("## 4. Relación features <-> target (Spearman, nivel landmark)")
L = pd.read_parquet("landmarks_full.parquet")
feats = [c for c in L.columns if c not in ["ID_COSECHA", "ID_TANDA", "D"]]
corr = {f: stats.spearmanr(L[f], L["D"], nan_policy="omit").statistic for f in feats}
cs = pd.Series(corr).dropna().sort_values()
log("Top 12 correlación NEGATIVA con la duración:")
log(cs.head(12).round(3).to_string() + "\n")
log("Top 12 correlación POSITIVA con la duración:")
log(cs.tail(12).round(3).to_string() + "\n")

topabs = cs.abs().sort_values(ascending=False).head(20).index
fig, ax = plt.subplots(figsize=(8, 6))
vals = cs[topabs].sort_values()
ax.barh(vals.index, vals.values, color=[ORANGE if v < 0 else BLUE for v in vals.values])
ax.set_title("Correlación de Spearman con la duración (top 20 |ρ|)")
ax.set_xlabel("ρ de Spearman"); plt.tight_layout(); plt.savefig("eda_corr_target.png", dpi=130); plt.close()

# scatters de las 4 más correlacionadas
fig, ax = plt.subplots(1, 4, figsize=(16, 4))
for i, f in enumerate(cs.abs().sort_values(ascending=False).head(4).index):
    s = L[[f, "D"]].dropna().sample(min(3000, len(L)), random_state=42)
    ax[i].scatter(s[f], s["D"], s=6, alpha=0.25, color=BLUE)
    ax[i].set_title(f"{f}  (ρ={cs[f]:.2f})"); ax[i].set_xlabel(f); ax[i].set_ylabel("duración")
plt.tight_layout(); plt.savefig("eda_scatter_target.png", dpi=130); plt.close()

with open("eda.md", "w") as f:
    f.write("\n".join(out))
    f.write("\n\n## Figuras\n- eda_target.png (distribución + estacionalidad)\n"
            "- eda_consistencia_temporal.png (drift por año, densidad, faltantes)\n"
            "- eda_train_vs_test.png (comparación de distribuciones)\n"
            "- eda_corr_target.png (correlaciones con el target)\n"
            "- eda_scatter_target.png (dispersión de las 4 más correlacionadas)\n")
log("\nGuardado: eda.md + 5 figuras PNG.")
