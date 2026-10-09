"""Extractor · CLIMA HISTÓRICO REAL (ERA5) — Open-Meteo Archive API.

Trae el clima que EFECTIVAMENTE ocurrió, día a día, para toda la ventana del
dataset de cosechas (con -60 días de colchón). Es la fuente para:
  - las features de clima PASADO del ciclo (días ya transcurridos), sin fuga;
  - la variante ORÁCULO del futuro (clima real futuro) como cota superior en el análisis.

Endpoint: https://archive-api.open-meteo.com/v1/archive  (ERA5, desde 1940, lag ~1-5 d)
Salida:   era5_real/clima_era5_diario.csv
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common as C

URL = "https://archive-api.open-meteo.com/v1/archive"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "clima_era5_diario.csv")


def main():
    start, end = C.rango_dataset(colchon_dias=60)
    print(f"ERA5 real · ventana {start} .. {end}")
    df = C.pull_daily(URL, {
        "latitude": C.LAT, "longitude": C.LON,
        "start_date": str(start), "end_date": str(end),
        "daily": C.DAILY_VARS, "timezone": C.TZ,
    })
    df.to_csv(OUT, index=False)
    print(f"Guardado: {OUT}  ({len(df)} filas, {df.shape[1]} columnas)")
    print("\n=== Clima medio por mes (sanity) ===")
    print(C.resumen_mensual(df))


if __name__ == "__main__":
    main()
