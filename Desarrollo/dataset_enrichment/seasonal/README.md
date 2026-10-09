# seasonal · Outlook estacional (~3-6 meses)

"Indicadores esperados para los próximos meses" (temperatura, precipitación) en vez de
por día. Sale del modelo estacional **SEAS5** como **ensemble de 50 miembros**, así que
da un **valor esperado** (media) y su **incertidumbre** (p10-p90).

- **API:** Open-Meteo Seasonal — `https://seasonal-api.open-meteo.com/v1/seasonal`
- **Horizonte:** ~6-7 meses hacia adelante. **Sin API key.**

## Correr
```bash
uv run seasonal/extractor.py
```
Salida: `snapshot_outlook_estacional.csv` — por mes y variable: `esperado`, `p10`, `p90`.

## ⚠️ Limitación clave para la tesis (verificado en vivo)

Hacia atrás **solo trae ~1 MES de datos** (la corrida vigente back-extendida: ~32 días,
2026-07-23 → 08-22). El parámetro `past_days=356` se acepta pero **devuelve NaN para
casi todo el pasado** — Open-Meteo seasonal solo sirve la corrida actual. **NO hay
archivo histórico gratis** y **no alcanza** ni para probar el aporte del estacional
(cubre apenas un puñado de cosechas, no un año). Sirve solo para **producción** (outlook
a futuro).

## Alternativas MEJORES para backfill (con hindcasts reales)

Si se quiere una feature estacional que **sí** se pueda reconstruir para 2022-2025 y
backtestear, hay dos fuentes con archivo de reforecasts que llegan mucho más atrás:

| Fuente | Hindcast desde | Qué da | Acceso | Peso |
|---|---|---|---|---|
| **NMME** (vía IRI Data Library) | 1982 | Anomalías mensuales y trimestrales de temp/precip, lead 1-11 meses, ensemble | URL OPeNDAP de `iridl.ldeo.columbia.edu` (devuelve datos de un punto) | Medio |
| **Copernicus C3S** (SEAS5 multi-modelo) | 1993 | Medias 1 y 3 meses de temp/precip, ensemble | CDS API (registro + GRIB/NetCDF) | Alto |

Recomendación: **NMME/IRI** es el camino más liviano para backfillear el horizonte
largo (se pide un punto lat/lon y devuelve series). Da temp y precipitación (no
radiación/DLI — limitación, porque el DLI es nuestra variable de luz más fuerte).

**Expectativa realista:** la habilidad de los pronósticos estacionales en la Patagonia
de latitud media es modesta (la señal de ENSO es débil ahí) y no incluyen luz. Puede que
no mejoren mucho sobre la climatología — pero al ser backtesteables, se puede **medir**
si aportan, que es lo correcto. Ver `../mvp_tiempo_cosecha/resumen.md` (backlog).
