"""Extractor · PRONÓSTICO EN VIVO — Open-Meteo Forecast API (para PRODUCCIÓN).

Este es el que se llamaría en tiempo real, cada día, para el ciclo en curso. Trae:
  - los últimos 7 días REALES (past_days) -> tapan el lag de ERA5 para los días recién
    transcurridos del ciclo;
  - los próximos 16 días de PRONÓSTICO -> features de clima futuro cercano.
Más allá de 16 días no hay pronóstico con habilidad: el tramo 17-60 se cubre con
climatología (ver README de la carpeta raíz).

No produce un dataset histórico: es una FOTO del pronóstico de hoy. Se guarda un
snapshot para inspección/demo; en producción el valor se consume en el momento.

Endpoint: https://api.open-meteo.com/v1/forecast
Salida:   pronostico_vivo/snapshot_pronostico_hoy.csv
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common as C

URL = "https://api.open-meteo.com/v1/forecast"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "snapshot_pronostico_hoy.csv")


def fetch_forecast(forecast_days: int = 16, past_days: int = 7):
    return C.pull_daily(URL, {
        "latitude": C.LAT, "longitude": C.LON,
        "daily": C.DAILY_VARS, "timezone": C.TZ,
        "forecast_days": min(forecast_days, 16), "past_days": past_days,
    })


def main():
    df = fetch_forecast()
    df.to_csv(OUT, index=False)
    print(f"Snapshot de pronóstico guardado: {OUT}  ({len(df)} filas)")
    cols = ["FECHA", "temperature_2m_mean", "dli_mol_m2", "cloud_cover_mean",
            "precipitation_sum"]
    print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
