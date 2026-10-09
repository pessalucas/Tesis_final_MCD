"""Objetivo 3 — rendimiento por cosecha (atribución tentativa).

La planilla de negocio 'HGP - Albahacas por cosecha.xlsx' tiene cantidad de plantas
cosechadas por día (2025-2026), NO por cosecha. Atribución tentativa:
  - Cada día de rendimiento se asigna a la cosecha cuyo cierre (COSECHA_FIN) está MÁS
    CERCA en el tiempo (no se excluye ningún día: los pedidos chicos se suman al cierre
    más cercano -> se conserva la masa total de plantas).
  - Se marca si la cosecha está 'aislada' (ningún otro cierre a ±2 días) para tener un
    subconjunto de etiquetas limpias.
Salida: rendimiento_por_cosecha.csv + sanity checks + figura.
SUPUESTO explícito: el volumen del entorno temporal pertenece a la cosecha que cierra
más cerca. Es tentativo (planilla de negocio, no 1:1 con el dataset principal).
"""
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import openpyxl

# --- rendimiento diario ---
wb = openpyxl.load_workbook("../dataset/HGP - Albahacas por cosecha.xlsx", read_only=True, data_only=True)
ws = wb["Hoja 1"]; rows = list(ws.iter_rows(values_only=True))[1:]
recs = []
for r in rows:
    if r[0] is not None: recs.append((r[0], r[1]))
    if len(r) > 4 and r[3] is not None: recs.append((r[3], r[4]))
d = pd.DataFrame(recs, columns=["fecha", "cant"])
d["fecha"] = pd.to_datetime(d["fecha"]); d["cant"] = pd.to_numeric(d["cant"], errors="coerce")
d = d.dropna().sort_values("fecha"); d = d[d["cant"] > 0].reset_index(drop=True)
total_planilla = d["cant"].sum()

# --- cierre por cosecha (dataset principal), solo 2025-2026 ---
main = pd.read_csv("../dataset/dataset_albahaca_1.csv", parse_dates=["FECHA"])
fin = main[main["COSECHA_FIN"] == 1]
cierre = fin.groupby("ID_COSECHA")["FECHA"].max().rename("cierre")  # último tanda que cierra
meta = main.groupby("ID_COSECHA").agg(
    inicio=("FECHA", "min"), cultivo=("CULTIVO", "first"),
    n_tandas=("ID_TANDA", "nunique")).join(cierre)
meta = meta[meta["cierre"].notna()]
meta["dur"] = (meta["cierre"] - meta["inicio"]).dt.days
meta["mes_cierre"] = meta["cierre"].dt.month
cos = meta[(meta["cierre"] >= "2025-01-01")].copy().sort_values("cierre")
print(f"Cosechas con cierre en 2025-2026: {len(cos)}")

# --- aislamiento (±2 días) ---
cd = cos["cierre"].dt.normalize().tolist()
cos["aislada"] = [all(abs((cd[j] - cd[i]).days) > 2 for j in range(len(cd)) if j != i) for i in range(len(cd))]

# --- atribución: cada día de rendimiento -> cosecha con cierre MÁS CERCANO ---
close_dates = cos["cierre"].dt.normalize().values.astype("datetime64[D]")
close_ids = cos.index.to_numpy()
asignado = []
for _, row in d.iterrows():
    fday = np.datetime64(row["fecha"].normalize(), "D")
    k = np.argmin(np.abs((close_dates - fday).astype(int)))
    dist = abs(int((close_dates[k] - fday).astype(int)))
    asignado.append((close_ids[k], dist))
d["cosecha"] = [a[0] for a in asignado]
d["dist_dias"] = [a[1] for a in asignado]

rend = d.groupby("cosecha").agg(rendimiento=("cant", "sum"), n_dias=("cant", "size"),
                                dist_media=("dist_dias", "mean")).reset_index()
cos = cos.merge(rend, left_index=True, right_on="cosecha", how="left").set_index("cosecha")
cos["rendimiento"] = cos["rendimiento"].fillna(0)

# --- sanity checks ---
print("\n=== SANITY ===")
print(f"Total plantas en planilla: {int(total_planilla)}")
print(f"Total atribuido a cosechas: {int(cos['rendimiento'].sum())}  (debe coincidir: conserva masa)")
print(f"Distancia media día->cierre asignado: {d['dist_dias'].mean():.1f} d | máx: {d['dist_dias'].max()} d")
print(f"Cosechas con rendimiento>0: {(cos['rendimiento']>0).sum()} de {len(cos)}")
print("\nDistribución del rendimiento por cosecha (plantas):")
print(cos["rendimiento"].describe().round(0).to_string())
print("\nRendimiento por cosecha — solo AISLADAS (etiqueta más limpia):")
print(cos.loc[cos["aislada"], "rendimiento"].describe().round(0).to_string())

# relación con duración y estación
print("\nCorrelación rendimiento vs duración (Spearman):",
      round(cos[["rendimiento", "dur"]].corr(method="spearman").iloc[0, 1], 3))
print("Rendimiento mediano por mes de cierre:")
print(cos.groupby("mes_cierre")["rendimiento"].median().round(0).to_string())

cos.to_csv("rendimiento_por_cosecha.csv")
print("\nGuardado: rendimiento_por_cosecha.csv")

# --- figura ---
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].hist(cos["rendimiento"], bins=25, color="#2f7de8", edgecolor="white")
ax[0].set_title("Rendimiento por cosecha (plantas)"); ax[0].set_xlabel("plantas")
ax[1].scatter(cos["dur"], cos["rendimiento"], s=18, alpha=0.6, color="#e8952f")
ax[1].set_title("Rendimiento vs duración"); ax[1].set_xlabel("duración (d)"); ax[1].set_ylabel("plantas")
cos.groupby("mes_cierre")["rendimiento"].median().plot(kind="bar", ax=ax[2], color="#2f7de8")
ax[2].set_title("Rendimiento mediano por mes de cierre"); ax[2].set_xlabel("mes")
plt.tight_layout(); plt.savefig("rendimiento_eda.png", dpi=130)
print("Guardado: rendimiento_eda.png")
