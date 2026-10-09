"""Builder · CLIMATOLOGÍA por día del año (normales) — derivado de ERA5.

No consume ninguna API: calcula, a partir del histórico real (era5_real), el clima
"típico" de cada día del año (promedio multi-anual, suavizado con ventana de ±7 días).

Para qué sirve: es el "clima esperado" para el tramo FUTURO largo del ciclo (más allá
de los ~16 días donde hay pronóstico con habilidad). Es leakage-free y 100%
reproducible en producción (no depende de saber el futuro real). Nota metodológica:
para rigor total, en validación cruzada la climatología debería calcularse solo con
los años de train; acá se usa global como primera aproximación (las normales son
estables, el aporte del propio año es despreciable).

Entrada: ../era5_real/clima_era5_diario.csv
Salida:  climatologia/climatologia_doy.csv  (366 filas, una por día del año)
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common as C

HERE = os.path.dirname(os.path.abspath(__file__))
IN = os.path.join(HERE, "..", "era5_real", "clima_era5_diario.csv")
OUT = os.path.join(HERE, "climatologia_doy.csv")
VENTANA = 7  # ±7 días de suavizado alrededor de cada día del año


def main():
    df = pd.read_csv(IN, parse_dates=["FECHA"])
    num = df.select_dtypes("number").columns.drop("doy", errors="ignore")
    # promedio por día del año
    clim = df.groupby("doy")[list(num)].mean()
    clim = clim.reindex(range(1, 367))
    # suavizado circular (envolver diciembre<->enero) con ventana ±VENTANA
    ext = pd.concat([clim, clim, clim])
    sm = ext.rolling(2 * VENTANA + 1, center=True, min_periods=1).mean().iloc[366:366 * 2]
    sm.index = clim.index
    sm.index.name = "doy"
    sm = sm.interpolate().bfill().ffill()
    sm.to_csv(OUT)
    print(f"Guardado: {OUT}  ({len(sm)} días del año, {sm.shape[1]} variables)")
    print("\nEjemplo — DLI y temperatura típicos:")
    muestra = sm.loc[[15, 105, 196, 288], ["temperature_2m_mean", "dli_mol_m2", "sunshine_h"]]
    muestra.index = ["15-ene (verano)", "15-abr (otoño)", "15-jul (invierno)", "15-oct (primavera)"]
    print(muestra.round(1))


if __name__ == "__main__":
    main()
