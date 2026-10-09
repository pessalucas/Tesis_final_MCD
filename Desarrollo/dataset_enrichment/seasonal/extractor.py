"""Extractor · OUTLOOK ESTACIONAL diario (Open-Meteo Seasonal API, SEAS5).

Pronóstico estacional como ENSEMBLE de 50 miembros -> por cada día y variable se
guarda el valor ESPERADO (media del ensemble) y su incertidumbre (p10-p90). Incluye
radiación solar -> se deriva el DLI esperado.

Guarda en el MISMO FORMATO diario que los demás datasets (una fila por FECHA), para
poder cruzar por fecha. Cubre:
  - tramo BACKTESTEABLE: solo ~1 MES (la corrida actual back-extendida, ~32 días).
    ⚠️ NO es un año: `past_days=356` se acepta pero devuelve NaN para casi todo el
    pasado. Open-Meteo seasonal solo sirve la corrida vigente. Insuficiente para probar
    el aporte del estacional (cubre apenas un puñado de cosechas).
  - tramo FORWARD: outlook a ~6 meses (producción).

⚠️ Límites: (1) NO hay archivo histórico gratis (solo ~1 mes de pasado); (2) el SEAS5
tiene días sin dato (resolución diaria irregular; su habilidad real es mensual);
(3) baja habilidad estacional en la Patagonia. Para un estacional backtesteable sobre
2022-2025 hace falta Meteomatics (programa académico gratis) o NMME/C3S (ver README).

Endpoint: https://seasonal-api.open-meteo.com/v1/seasonal
Salida:   seasonal/seasonal_diario.csv
"""
import datetime as dt
import os
import sys

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common as C

URL = "https://seasonal-api.open-meteo.com/v1/seasonal"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "seasonal_diario.csv")
BASES = ["temperature_2m_max", "temperature_2m_min", "precipitation_sum",
         "shortwave_radiation_sum"]


def main():
    r = requests.get(URL, params={
        "latitude": C.LAT, "longitude": C.LON,
        "daily": BASES, "past_days": 356, "forecast_days": 214, "timezone": C.TZ,
    }, timeout=180)
    r.raise_for_status()
    d = r.json()["daily"]
    df = pd.DataFrame(d).rename(columns={"time": "FECHA"})
    df["FECHA"] = pd.to_datetime(df["FECHA"])

    out = pd.DataFrame({"FECHA": df["FECHA"]})
    for base in BASES:
        mem = [c for c in df.columns if c.startswith(base + "_member")]
        M = df[mem].to_numpy(float)
        with np.errstate(invalid="ignore"):
            out[f"{base}__esperado"] = np.nanmean(M, axis=1)
            out[f"{base}__p10"] = np.nanpercentile(M, 10, axis=1)
            out[f"{base}__p90"] = np.nanpercentile(M, 90, axis=1)
    # DLI esperado (mol/m²) de la radiación esperada
    out["dli__esperado"] = out["shortwave_radiation_sum__esperado"] * 2.04

    # marcar el tramo backtesteable (pasado) vs forward (producción)
    hoy = pd.Timestamp(dt.date.today())
    out["es_backtesteable"] = (out["FECHA"] <= hoy).astype(int)

    # quedarse solo con días que tienen dato (el SEAS5 tiene huecos)
    out = out.dropna(subset=[f"{BASES[0]}__esperado"]).reset_index(drop=True)
    out.to_csv(OUT, index=False)

    past = out[out["es_backtesteable"] == 1]
    print(f"Guardado: {OUT}  ({len(out)} días con dato, {out.shape[1]-1} columnas)")
    print(f"Tramo backtesteable: {past['FECHA'].min().date()} -> {past['FECHA'].max().date()} "
          f"({len(past)} días)")
    print(f"Tramo forward: {out['FECHA'].max().date()} (outlook producción)")
    print("\nEjemplo (temp max esperada [p10-p90], DLI esperado):")
    m = out.iloc[::max(1, len(out)//6)][
        ["FECHA", "temperature_2m_max__esperado", "temperature_2m_max__p10",
         "temperature_2m_max__p90", "dli__esperado"]]
    print(m.to_string(index=False))


if __name__ == "__main__":
    main()
