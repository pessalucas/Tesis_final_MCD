# Enrichment climático — invernadero de Bariloche

Datos climáticos para enriquecer los modelos de predicción del tiempo de cosecha.
En un invernadero de la Patagonia el clima es casi con seguridad **el mecanismo físico
detrás de la fuerte estacionalidad** de la duración de los ciclos (verano ~37 días,
invierno ~60). Chequeo de sanidad confirmado: la curva de luz/temperatura de Bariloche
calca la de duración.

> Contexto del proyecto y backlog: `../mvp_tiempo_cosecha/resumen.md`. Este enrichment
> es el experimento **EXP-D1**.

## Ubicación
Invernadero **Con Con 7900, San Carlos de Bariloche, Río Negro**.
Coordenadas: **lat −41.1078158, lon −71.1945077**. Huso `America/Argentina/Salta`.

## Organización — un dataset por fuente/API

Cada carpeta es autoexplicativa (extractor + CSV + README propio). Config y variables
compartidas en `common.py`.

| Carpeta | Fuente / API | Qué es | Rol |
|---|---|---|---|
| **`era5_real/`** | Archive API (ERA5) | Clima que **ocurrió**, diario | Clima pasado del ciclo (sin fuga) + oráculo del futuro |
| **`pronostico_lead/`** | Previous Runs API | Pronóstico pasado **con error real**, lead 1-7 d | Feature de pronóstico cercano, **backtesteable** |
| **`climatologia/`** | derivado de ERA5 (sin API) | Normales por día del año | Clima esperado del futuro lejano (leakage-free) |
| **`pronostico_vivo/`** | Forecast API | Pronóstico 16 d en vivo | **Producción** (tiempo real) |
| **`seasonal/`** | Seasonal API (SEAS5) | Outlook 3-6 meses, ensemble | Solo producción (backtesteable apenas ~1 mes, no sirve para training) |

> Nota: se probó también la **Historical Forecast API** y para esta ubicación resultó
> **idéntica a ERA5** (sin error de pronóstico) → descartada. La API correcta para
> pronósticos pasados es **Previous Runs** (`pronostico_lead/`).

## Cómo correr todo
```bash
# entorno uv autocontenido (Python 3.12). Desde esta carpeta:
uv run era5_real/extractor.py
uv run pronostico_lead/extractor.py          # ~1-2 min (baja horario por chunks de año)
uv run climatologia/build_climatologia.py    # requiere era5_real ya generado
uv run pronostico_vivo/extractor.py          # snapshot del pronóstico de hoy
uv run seasonal/extractor.py                 # outlook estacional a 6 meses
```

## Esquema (columnas comunes a los CSV diarios)

Grano: una fila por día. Join contra las cosechas por **`FECHA`**.

**Crudas:** temperatura (`temperature_2m_max/min/mean`, `apparent_temperature_*`),
luz (`sunshine_duration`, `daylight_duration`, `shortwave_radiation_sum` MJ/m²),
nubosidad (`cloud_cover_mean/max/min`), agua (`precipitation_sum`, `rain_sum`,
`snowfall_sum`, `precipitation_hours`, `et0_fao_evapotranspiration`), humedad
(`relative_humidity_2m_mean/max/min`, `dew_point_2m_mean`, `vapour_pressure_deficit_max`),
presión (`pressure_msl_mean`, `surface_pressure_mean`), viento (`wind_speed_10m_max`,
`wind_gusts_10m_max`, `wind_direction_10m_dominant`).

**Derivadas** (`common.add_derived`): `dli_mol_m2` (Daily Light Integral, la métrica de
luz para plantas), `sunshine_h`, `daylight_h`, `temp_amplitud`, `gdd_base10`
(grados-día albahaca), `doy`.

## Diseño anti-leakage (clave metodológica de la tesis)

Para predecir, el día `d` de un ciclo, cuántos días faltan hasta la cosecha, separar:

1. **Clima PASADO real** (días 1..d + ventana −60d) → `era5_real`. Sin fuga. Grueso de
   la señal.
2. **Clima FUTURO realista** (días d+1..cosecha):
   - cercano (≤7 d): pronóstico pasado real → `pronostico_lead` (backtesteable en
     training; en producción, `pronostico_vivo`);
   - lejano (8-60 d): `climatologia` (en producción se puede sumar `seasonal`).
3. **Clima FUTURO oráculo** (real futuro, `era5_real`): **solo** como cota superior en
   el análisis. Nunca en producción. La brecha oráculo − realista mide el costo de la
   incertidumbre del pronóstico.

**Hallazgos sobre el pronóstico pasado:**
- La **Historical Forecast API** para esta ubicación resultó idéntica a ERA5 (no hay
  error de pronóstico) → descartada.
- La **Previous Runs API** (`pronostico_lead/`) sí da pronóstico pasado con error real,
  lead 1-7 días, cubriendo todo el dataset → es la feature de pronóstico backtesteable.
- El horizonte **largo** (semanas-meses) no es backtesteable con Open-Meteo (Seasonal
  solo desde 2025-09); para eso hacen falta hindcasts de NMME/IRI o Copernicus C3S.

## Producción (tiempo real)
1. Cada día: `pronostico_vivo` (7 días reales + 16 de pronóstico).
2. Features de pasado con los reales; de futuro cercano con el pronóstico; de futuro
   lejano con `climatologia`.
3. Mismo formato que en entrenamiento → sin train/serve skew. 1 request/día. Sin key.

## Futuros enrichments (guía para el próximo agente)
- **Nueva variable**: agregar a `DAILY_VARS` en `common.py` (validar que exista como
  *daily* en la API) y derivar en `add_derived()`. Re-correr.
- **Otra ubicación**: cambiar `LAT`, `LON`, `TZ` en `common.py`. El resto es automático.
- **Cross-check del histórico**: NASA POWER (`power.larc.nasa.gov`), gratis, sin
  restricción comercial, fuerte en radiación solar; sin pronóstico.
- **Backfill estacional real (hindcasts)**: para tener una feature de outlook a 1-3
  meses backtesteable en 2022-2025, agregar una carpeta que baje **NMME** (vía IRI Data
  Library, `iridl.ldeo.columbia.edu`, hindcasts desde 1982, punto lat/lon vía OPeNDAP) o
  **Copernicus C3S** (CDS API, hindcasts desde 1993, GRIB). Dan temp/precip, no
  radiación. Ojo: baja habilidad en la Patagonia; medir si aportan sobre climatología.
- **Cruce con cosechas**: por `FECHA`. Las features de ventana (pasado del ciclo, −60d,
  futuro) se calculan en el paso de modelado, no acá.
