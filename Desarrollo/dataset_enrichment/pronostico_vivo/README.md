# pronostico_vivo · Pronóstico en vivo (para PRODUCCIÓN)

El endpoint que se llamaría **en tiempo real**, cada día, para el ciclo en curso.
No genera un dataset histórico: es una **foto** del pronóstico de hoy.

- **API:** Open-Meteo Forecast — `https://api.open-meteo.com/v1/forecast`
- **Trae:** últimos 7 días REALES (`past_days=7`, tapan el lag de ERA5) + próximos 16
  días de PRONÓSTICO (`forecast_days=16`).
- **Sin API key.**

## Correr
```bash
uv run pronostico_vivo/extractor.py
```
Salida: `snapshot_pronostico_hoy.csv` (inspección/demo). En producción el valor se
consume en el momento, no se persiste.

## Rol en producción
1. Cada día, para el ciclo en curso, llamar a `fetch_forecast()`.
2. Clima pasado del ciclo → de los `past_days` reales (o de ERA5 con lag).
3. Clima futuro cercano (≤16 d) → del pronóstico.
4. Clima futuro lejano (17-60 d) → de `climatologia`.
5. Mismo formato y variables que en entrenamiento → sin train/serve skew.

Costo: 1 request por día por invernadero. Trivial.

## Único lugar donde el pronóstico REAL aporta señal
A diferencia de `pronostico_archivado` (que resultó idéntico a ERA5 para el pasado),
acá el pronóstico **sí** difiere de lo que finalmente ocurra: es la anomalía a corto
plazo ("se vienen 2 semanas más cálidas/soleadas que lo normal"). Ese es el aporte
predictivo incremental de un pronóstico, y solo es observable hacia adelante. Para
cuantificarlo con rigor habría que **registrar el pronóstico y compararlo luego** con
lo ocurrido, sobre ciclos nuevos.
