# climatologia · Normales climáticas por día del año

Clima **típico** de cada día del año en Bariloche (promedio multi-anual suavizado),
derivado del histórico real. **No consume API.**

- **Entrada:** `../era5_real/clima_era5_diario.csv`
- **Método:** promedio por día del año (`doy`) sobre 2022-2026, suavizado circular ±7
  días (envuelve diciembre↔enero).

## Correr
```bash
uv run climatologia/build_climatologia.py   # requiere era5_real ya generado
```
Salida: `climatologia_doy.csv` — 366 filas (una por día del año), 32 variables.

## Para qué sirve
Es el **"clima esperado" del tramo futuro** del ciclo, sobre todo más allá de los ~16
días donde el pronóstico tiene habilidad. Ventajas:
- **Leakage-free**: no usa el futuro real; solo el calendario.
- **Reproducible en producción**: se calcula una vez y se consulta por `doy`.

Ejemplo (típicos): 15-ene DLI 63 / 17.5 °C · 15-jul DLI 13 / 2 °C. La estacionalidad
que explica los ciclos cortos de verano y largos de invierno.

## Nota metodológica
El aporte incremental de la climatología del futuro sobre las features de fecha
(`mes`, `doy`) es limitado: la climatología es casi una función determinística del
calendario, que el modelo ya infiere. Su utilidad real es para construir **anomalías**
(desvío del pronóstico/observación respecto de lo normal) y para tener un futuro
esperado explícito y consistente.

Para rigor total en validación cruzada, calcular las normales solo con los años de
train. Acá se usan globales como primera aproximación (las normales son estables).
