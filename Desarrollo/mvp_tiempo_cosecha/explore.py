"""Exploración del dataset de hidroponía antes de modelar.
Objetivo: entender la unidad de análisis (tanda), duraciones, censura,
calidad de datos y disponibilidad de features. Nada de modelado todavía.
"""
import pandas as pd
import numpy as np

CSV = "../dataset/dataset_albahaca_1.csv"

df = pd.read_csv(CSV, parse_dates=["FECHA"])
# Normalizar PPM a una sola escala (mS/cm -> ppm, factor 500 estándar)
df["PPM"] = np.where(df["UNIDAD"].eq("mS/cm"), df["PPM"] * 500, df["PPM"])
print("CULTIVO:", df["CULTIVO"].value_counts().to_dict())
print("Filas totales:", len(df))
print("Columnas:", list(df.columns))
print("HOJA únicas:", df["HOJA"].unique())
print("UNIDAD únicas:", df["UNIDAD"].unique())

# Unidad de análisis = (ID_COSECHA, ID_TANDA)
g = df.groupby(["ID_COSECHA", "ID_TANDA"])
print("\n# cosechas:", df["ID_COSECHA"].nunique())
print("# tandas (cosecha x tanda):", g.ngroups)

# ¿Cuántas tandas tienen COSECHA_FIN? (evento observado = duración conocida)
fin = g["COSECHA_FIN"].max()
print("\nTandas con COSECHA_FIN=1 (completas):", int((fin == 1).sum()))
print("Tandas sin cierre (censuradas):", int((fin == 0).sum()))

# Duración = DIA_DESDE_INICIO en el día de COSECHA_FIN, para tandas completas
def duracion_tanda(sub):
    fin_rows = sub[sub["COSECHA_FIN"] == 1]
    if len(fin_rows) == 0:
        return pd.Series({"completa": 0, "duracion": np.nan,
                          "n_dias_obs": sub["DIA_DESDE_INICIO"].max(),
                          "cultivo": sub["CULTIVO"].iloc[0],
                          "fraccion": str(sub["FRACCION"].iloc[0]),
                          "mes_inicio": sub["FECHA"].min().month})
    # día del evento (primer COSECHA_FIN)
    dur = fin_rows["DIA_DESDE_INICIO"].min()
    return pd.Series({"completa": 1, "duracion": dur,
                      "n_dias_obs": sub["DIA_DESDE_INICIO"].max(),
                      "cultivo": sub["CULTIVO"].iloc[0],
                      "fraccion": str(sub["FRACCION"].iloc[0]),
                      "mes_inicio": sub["FECHA"].min().month})

summ = g.apply(duracion_tanda, include_groups=False).reset_index()
comp = summ[summ["completa"] == 1]
print("\n=== Duración (días) de tandas COMPLETAS ===")
print(comp["duracion"].describe())
print("\nDuración por cultivo (mediana / n):")
print(comp.groupby("cultivo")["duracion"].agg(["count", "median", "mean", "std", "min", "max"]))

print("\n=== Duración por mes de inicio (estacionalidad) ===")
print(comp.groupby("mes_inicio")["duracion"].agg(["count", "median"]))

# Cosechas independientes que aportan tandas completas
print("\nCosechas con >=1 tanda completa:", comp["ID_COSECHA"].nunique())
print("Tandas completas por cosecha (describe):")
print(comp.groupby("ID_COSECHA").size().describe())

# Calidad: faltantes de PH/PPM
print("\nFaltantes PH:", df["PH"].isna().sum(), "| PPM:", df["PPM"].isna().sum())
print("Rango PH:", df["PH"].min(), df["PH"].max())
print("Rango PPM:", df["PPM"].min(), df["PPM"].max())

# Eventos disponibles como señal
eventos = ["TRASPASO_TANQUE","AGREGADO_SALES","CAMBIO_TANQUE","CAMBIO_80_TANQUE",
           "BOVEL","ESTEN_80","G_A_F1"]
print("\nEventos (suma):")
print(df[eventos].sum())

# ¿La serie por tanda es contigua día a día? chequeo rápido
lens = comp["duracion"]
print("\nSi construyo landmarks (un sample por día observado, día>=3):")
print("total samples aprox:", int((comp["duracion"].clip(lower=3) - 2).sum()))
