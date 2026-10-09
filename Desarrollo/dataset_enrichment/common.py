"""Configuración y utilidades compartidas por los extractores de clima.

Cada fuente de datos (API distinta) vive en su propia subcarpeta con su extractor,
su CSV y su README. Todas comparten: la ubicación del invernadero, el set de
variables diarias y las derivadas. Eso se centraliza acá para no duplicar.
"""
from __future__ import annotations
import datetime as dt
import os
import numpy as np
import pandas as pd
import requests

# --------------------------------------------------------------------------- #
# Ubicación del invernadero (Con Con 7900, San Carlos de Bariloche, Río Negro).
# Coordenadas del pin de Google Maps.
# --------------------------------------------------------------------------- #
LAT, LON = -41.1078158, -71.1945077
TZ = "America/Argentina/Salta"          # UTC-3, sin horario de verano

ROOT = os.path.dirname(os.path.abspath(__file__))
CSV_HARVEST = os.path.join(ROOT, "..", "dataset", "dataset_albahaca_1.csv")

# Variables diarias comunes a todas las fuentes (validadas contra la API).
DAILY_VARS = [
    "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
    "apparent_temperature_max", "apparent_temperature_min", "apparent_temperature_mean",
    "sunshine_duration", "daylight_duration", "shortwave_radiation_sum",
    "cloud_cover_mean", "cloud_cover_max", "cloud_cover_min",
    "precipitation_sum", "rain_sum", "snowfall_sum", "precipitation_hours",
    "et0_fao_evapotranspiration",
    "relative_humidity_2m_mean", "relative_humidity_2m_max", "relative_humidity_2m_min",
    "dew_point_2m_mean", "vapour_pressure_deficit_max",
    "pressure_msl_mean", "surface_pressure_mean",
    "wind_speed_10m_max", "wind_gusts_10m_max", "wind_direction_10m_dominant",
]


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Variables derivadas relevantes para el cultivo."""
    if "shortwave_radiation_sum" in df:
        # DLI (Daily Light Integral, mol PAR/m²/día): métrica de luz para plantas.
        # PAR ~45% de la energía de onda corta; 1 MJ PAR ~4.57 mol -> factor ~2.04.
        df["dli_mol_m2"] = df["shortwave_radiation_sum"] * 2.04
    if "sunshine_duration" in df:
        df["sunshine_h"] = df["sunshine_duration"] / 3600
    if "daylight_duration" in df:
        df["daylight_h"] = df["daylight_duration"] / 3600
    if {"temperature_2m_max", "temperature_2m_min"} <= set(df):
        df["temp_amplitud"] = df["temperature_2m_max"] - df["temperature_2m_min"]
    if "temperature_2m_mean" in df:
        df["gdd_base10"] = (df["temperature_2m_mean"] - 10).clip(lower=0)  # albahaca, base 10 °C
    df["doy"] = df["FECHA"].dt.dayofyear
    return df


def pull_daily(url: str, params: dict) -> pd.DataFrame:
    """GET a un endpoint diario de Open-Meteo -> DataFrame con FECHA + derivadas."""
    r = requests.get(url, params=params, timeout=120)
    if r.status_code == 400:
        raise SystemExit(f"API 400: {r.json().get('reason', r.text)}")
    r.raise_for_status()
    data = r.json()
    if "error" in data:
        raise SystemExit(f"API error: {data}")
    df = pd.DataFrame(data["daily"]).rename(columns={"time": "FECHA"})
    df["FECHA"] = pd.to_datetime(df["FECHA"])
    return add_derived(df)


def rango_dataset(colchon_dias: int = 60) -> tuple[dt.date, dt.date]:
    """Rango de fechas del dataset de cosechas, con `colchon_dias` antes del inicio
    (para poder calcular ventanas previas). Capea el fin a ayer (sin futuro)."""
    h = pd.read_csv(CSV_HARVEST, usecols=["FECHA"], parse_dates=["FECHA"])
    start = (h["FECHA"].min() - pd.Timedelta(days=colchon_dias)).date()
    end = min(h["FECHA"].max().date(), dt.date.today() - dt.timedelta(days=1))
    return start, end


def resumen_mensual(df: pd.DataFrame) -> pd.DataFrame:
    """Chequeo de sanidad: clima medio por mes (debe explicar la estacionalidad)."""
    df = df.copy()
    df["mes"] = df["FECHA"].dt.month
    return df.groupby("mes").agg(
        t_media=("temperature_2m_mean", "mean"),
        dli=("dli_mol_m2", "mean"),
        sol_h=("sunshine_h", "mean"),
        nubosidad=("cloud_cover_mean", "mean"),
        lluvia=("precipitation_sum", "mean"),
    ).round(1)
