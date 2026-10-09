"""Extractor · PRONÓSTICO PASADO con error real (Open-Meteo Previous Runs API).

Esta es la API correcta para recuperar "el pronóstico tal como se emitió" en fechas
pasadas (a diferencia de Historical Forecast, que devolvía = ERA5). Da el valor
pronosticado para cada día a LEAD FIJO de 1 a 7 días (variables horarias
`*_previous_dayN`). Cobertura hacia atrás: >= 2021 -> cubre todo el dataset.

Construye la vista por FECHA DE EMISIÓN (issue-date), que es la que usa el modelo:
para el día de predicción F, el pronóstico a horizonte k (1..7) es el valor
`previous_dayk` válido para la fecha F+k. Sin fuga: es exactamente lo que se sabía el
día F, reproducible igual en producción.

Endpoint: https://previous-runs-api.open-meteo.com/v1/forecast
Salida:   pronostico_lead/pronostico_lead_por_emision.csv  (una fila por fecha de emisión)
"""
import os
import sys

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common as C

URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pronostico_lead_por_emision.csv")
LEADS = range(1, 8)   # 1..7 días (máx disponible)

# base -> (cómo agregar horario->diario, factor de escala para pasar a unidad diaria)
#   temperatura/nubosidad/humedad: media diaria
#   radiación: los horarios son W/m²; MJ/día = sum(W/m²)*3600/1e6 = sum*0.0036
#   precipitación: suma diaria
BASES = {
    "temperature_2m":       ("mean", 1.0),
    "shortwave_radiation":  ("sum",  0.0036),   # -> MJ/m²/día
    "cloud_cover":          ("mean", 1.0),
    "relative_humidity_2m": ("mean", 1.0),
    "precipitation":        ("sum",  1.0),
}


def pull_year(y0, y1):
    hv = [f"{b}_previous_day{n}" for b in BASES for n in LEADS]
    r = requests.get(URL, params={
        "latitude": C.LAT, "longitude": C.LON,
        "start_date": str(y0), "end_date": str(y1),
        "hourly": hv, "timezone": C.TZ,
    }, timeout=180)
    if r.status_code == 400:
        raise SystemExit(f"API 400: {r.json().get('reason')}")
    r.raise_for_status()
    h = pd.DataFrame(r.json()["hourly"])
    h["time"] = pd.to_datetime(h["time"])
    return h.set_index("time")


def main():
    start, end = C.rango_dataset(colchon_dias=0)   # el pronóstico no necesita colchón previo
    print(f"Previous Runs · ventana {start} .. {end} (lead 1-7 d)")

    # pull por año para no pedir un JSON gigante de una
    chunks = []
    for y in range(start.year, end.year + 1):
        a = max(start, pd.Timestamp(f"{y}-01-01").date())
        b = min(end, pd.Timestamp(f"{y}-12-31").date())
        print(f"  bajando {a} .. {b} ...")
        chunks.append(pull_year(a, b))
    hourly = pd.concat(chunks)

    # horario -> diario, por (base, lead), en la VISTA POR FECHA VÁLIDA
    daily_valid = {}
    for base, (agg, scale) in BASES.items():
        for n in LEADS:
            col = f"{base}_previous_day{n}"
            s = getattr(hourly[col].resample("D"), agg)() * scale
            daily_valid[(base, n)] = s

    # reensamblar a VISTA POR FECHA DE EMISIÓN F:
    #   pronóstico emitido en F para F+k  =  daily_valid[base,k] en la fecha F+k
    #   -> shift(-k) lleva el valor de F+k a la fila F
    idx = pd.date_range(start, end, freq="D")
    out = pd.DataFrame(index=idx)
    for base, (agg, scale) in BASES.items():
        cols_horizonte = []
        for n in LEADS:
            s = daily_valid[(base, n)].shift(-n).reindex(idx)
            out[f"{base}__d{n}"] = s
            cols_horizonte.append(f"{base}__d{n}")
        # resumen "próximos 7 días": media (o suma para precipitación)
        if agg == "sum":
            out[f"{base}__next7"] = out[cols_horizonte].sum(axis=1, min_count=1)
        else:
            out[f"{base}__next7"] = out[cols_horizonte].mean(axis=1)

    # DLI pronosticado (mol/m²) a partir de la radiación pronosticada
    for n in LEADS:
        out[f"dli__d{n}"] = out[f"shortwave_radiation__d{n}"] * 2.04
    out["dli__next7"] = out["shortwave_radiation__next7"] * 2.04

    out.index.name = "FECHA"
    out = out.reset_index()
    out.to_csv(OUT, index=False)
    print(f"\nGuardado: {OUT}  ({len(out)} fechas de emisión, {out.shape[1]-1} columnas de pronóstico)")

    # sanity: el pronóstico a 1 día debería correlacionar fuerte con el real; a 7 días, menos
    real = pd.read_csv(os.path.join(C.ROOT, "era5_real", "clima_era5_diario.csv"),
                       parse_dates=["FECHA"]).set_index("FECHA")["temperature_2m_mean"]
    print("\n=== Sanity: correlación pronóstico de temp vs real, por lead ===")
    print("(emitido el día F para F+k, comparado con la temp real de F+k)")
    for n in LEADS:
        s = daily_valid[("temperature_2m", n)].reindex(real.index)
        c = s.corr(real)
        err = (s - real).abs().mean()
        print(f"  lead {n} d: corr={c:.3f}  MAE={err:.2f} °C")


if __name__ == "__main__":
    main()
