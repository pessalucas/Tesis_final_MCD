# era5_real · Clima histórico REAL (ERA5)

Clima que **efectivamente ocurrió**, día a día, en Bariloche. Fuente principal del
enrichment.

- **API:** Open-Meteo Archive — `https://archive-api.open-meteo.com/v1/archive`
- **Producto:** reanálisis ERA5 (desde 1940; lag ~1-5 días respecto de hoy).
- **Sin API key** (uso no comercial / tesis).

## Correr
```bash
uv run era5_real/extractor.py   # desde la carpeta dataset_enrichment/
```
Un solo request cubre todo el rango. Rango calculado solo desde el dataset de
cosechas, con −60 días de colchón (para ventanas previas).

## Salida
`clima_era5_diario.csv` — una fila por día (`FECHA`), 34 columnas.
Rango: **2022-05-04 → 2026-08-22** (1.572 días). Variables y derivadas: ver el README
raíz (`../README.md`), sección "Esquema".

## Uso en el modelo
- **Clima PASADO del ciclo** (días 1..d ya transcurridos + ventana −60d): sin fuga,
  disponible el día d. Es el grueso de la señal usable.
- **Variante ORÁCULO del futuro** (clima real de los días d+1..cosecha): SOLO como cota
  superior en el análisis (cuánto se ganaría con pronóstico perfecto). Nunca como
  feature de producción — sería leakage.

## Notas
- Grilla ERA5 ~9-25 km: capta la estacionalidad regional, no microclima del predio.
- La cola futura del dataset (2026-08-23 → 09-04) no está en ERA5; en producción se
  cubre con `pronostico_vivo`.
