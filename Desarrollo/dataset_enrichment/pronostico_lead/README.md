# pronostico_lead · Pronóstico pasado con error real (Previous Runs API)

**La API correcta para "pronósticos pasados".** A diferencia de la Historical Forecast
API (que devolvía = ERA5, sin error), esta da el valor que el modelo **pronosticaba**
para cada día a **lead fijo de 1 a 7 días**, con su error real. Cobertura hacia atrás
≥ 2021 → **cubre todo el dataset**. Es una feature **backtesteable** (no solo de
producción).

- **API:** Open-Meteo Previous Runs — `https://previous-runs-api.open-meteo.com/v1/forecast`
- **Variables:** horarias `*_previous_dayN` (N=1..7). Sin API key.

## Qué genera

Vista **por fecha de emisión** (issue-date), que es la que usa el modelo: para el día de
predicción `F`, el pronóstico a horizonte `k` (1..7 días) tal como se conocía ese día.
Reconstrucción sin fuga: `pronóstico(F → F+k) = previous_day k` válido para `F+k`.

Salida: `pronostico_lead_por_emision.csv` — una fila por fecha de emisión `F`
(1.512 fechas), 48 columnas. Variables: temperatura, radiación→DLI, nubosidad, humedad,
precipitación. Por cada una: `{var}__d1..d7` (cada horizonte) y `{var}__next7` (resumen
próximos 7 días; suma para precipitación, media para el resto).

## Correr
```bash
uv run pronostico_lead/extractor.py   # baja por chunks de año (~1-2 min)
```

## Sanity check (confirmado)

Correlación y error del pronóstico de temperatura vs. lo que realmente ocurrió, por lead:

| Lead | Correlación | MAE |
|---|---|---|
| 1 día | 0.986 | 0.79 °C |
| 3 días | 0.981 | 0.90 °C |
| 5 días | 0.967 | 1.22 °C |
| 7 días | 0.934 | 1.70 °C |

**El error crece con el lead** — justo lo esperable de un pronóstico real. Si esto fuera
ERA5 disfrazado (como el `historical-forecast`), el MAE sería 0. Prueba de que es
pronóstico genuino con incertidumbre.

## Uso en el modelo (sin fuga)
Cruzar por `FECHA` = día de predicción. Estas features aportan la **anomalía a corto
plazo** ("se vienen 7 días más cálidos/soleados que lo normal"), con más valor cerca de
la cosecha, donde el tramo restante entra dentro del horizonte de 7 días. Más allá de 7
días: usar `climatologia` (y, para producción, `seasonal`).
