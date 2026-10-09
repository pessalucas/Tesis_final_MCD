# Resumen de experimentos — Predicción del tiempo de cosecha (tesis hidroponía)

Documento vivo. Bitácora de lo que ya probamos + backlog de lo que vamos a testear.
La idea es acumular evidencia para una tesis rica: distintos framings del problema,
distintos modelos, distintas features, y una lectura honesta de cada uno.

> **Eje central de la tesis:** predecir el tiempo de cosecha como problema de
> *tiempo-al-evento*, re-estimable día a día. Aristas de apoyo: germinación,
> rendimiento, factores accionables.

---

## 0. Setup común (aplica a todos los experimentos salvo que se diga)

- **Dataset:** `../dataset/dataset_albahaca_1.csv` — 2 hojas, solo albahaca.
  211 cosechas, 630 tandas, 567 completas, jul-2022 → sep-2026.
- **Unidad de análisis:** la tanda `(ID_COSECHA, ID_TANDA)`. Evento = `COSECHA_FIN`.
- **Unidad de generalización (clave):** la **cosecha** (`ID_COSECHA`). Hay **186
  cosechas independientes** con ≥1 tanda completa. Ese es el n real, no las ~25k filas.
- **Validación:** `GroupKFold(5)` agrupando por `ID_COSECHA`. Nunca mezclar días ni
  tandas de una misma cosecha entre train y test. Reportar spread entre folds.
- **Métrica:** MAE en días sobre la duración total `D` (≡ días restantes, porque el
  día actual se conoce exacto). Complementar con MedAE y, más adelante, métricas de
  supervivencia (C-index, Brier) y de calibración.
- **Baseline obligatorio (dummy):** `D_hat = duración media del train`. Todo modelo
  se compara contra esto.
- **Limpieza:** normalizar PPM (mS/cm → ppm, **factor 500 SUPUESTO**, confirmar si es
  640); pH fuera de [0,14] → NaN (ej. T59/FC534=1620).
- **Entorno:** `uv` con Python 3.12. Correr con `uv run <script>.py`.
  (El Python del sistema no funciona sin Xcode tools.)

### Archivos
| Archivo | Qué hace |
|---|---|
| `explore.py` | EDA: duraciones, censura, estacionalidad, calidad. |
| `model.py` | Pipeline landmark + 3 modelos + FI + gráficos. Guarda `landmarks.parquet`. |
| `events_probe.py` | Sondas sobre el aporte de los eventos. |
| `mae_por_etapa.png`, `feature_importance.png` | Gráficos. |

---

## 1. Estado actual — qué ya probamos

### EXP-01 · Landmark regression (MVP baseline) ✅ HECHO
- **Framing:** un ejemplo por tanda × día observado (features solo del pasado),
  target = `D`. Un solo modelo que se re-consulta cada día → se corrige con el tiempo.
- **Modelo:** `HistGradientBoostingRegressor` (maneja NaN nativo, no-linealidades,
  regularizado: depth 4, min_leaf 40, l2 1.0).
- **Features:** estáticas (mes cíclico, cultivo, fracción, día actual) + trayectoria
  (pH/PPM: last/mean/std/min/max/slope/mean7; estado de tanque; eventos: conteo y
  `dias_desde_`).
- **Resultado (OOF):**

  | Modelo | MAE (d) | vs dummy |
  |---|---|---|
  | Dummy (D=promedio) | 8.19 | — |
  | Solo estáticas | 4.42 | −46 % |
  | **Completo (+trayectoria)** | **3.20** | **−61 %** |

  Aporte de la trayectoria: **+27.6 %** sobre solo-estáticas. Estable entre folds
  (2.89–3.73). Mejor error hacia el día 21–30 (~3.0 d).
- **Lectura:** viabilidad confirmada. La época del año domina; la trayectoria refina y
  su aporte crece con el ciclo.

### Hallazgos transversales ya establecidos
- **Tamaño de muestra manda:** con 89 cosechas (dataset viejo) la trayectoria aportaba
  ~6 %; con 186 saltó a 27.6 %. Evidencia directa del argumento central de la tesis.
- **Los eventos SÍ aportan:** solos bajan MAE de 8.18 → 5.17. El "≈0" en permutation
  importance de varios era **redundancia** (con día actual / estación / PPM), no falta
  de señal. Para grupos correlacionados hay que medir aporte del **grupo** (drop-in),
  no feature por feature. `dias_desde_TRASPASO_TANQUE` y `n_BOVEL` están en el top.
- **Recencia > conteo** para `G_A_F1` y `AGREGADO_SALES`; conteo > recencia para `BOVEL`.

---

## 2. Backlog de experimentos

Marcar: ⬜ pendiente · 🔄 en curso · ✅ hecho. Cada uno va con su entrada en la
bitácora (sección 3) cuando se corre.

### A. Framings del problema (el corazón de la tesis)
- ✅ **EXP-A1 · Supervivencia (tiempo-al-evento).** Random Survival Forest sobre
  landmarks, usando las 62 tandas censuradas. **C-index 0.896** (excelente
  discriminación). Pero MAE del tiempo restante esperado (RMST) 3.83 d, PEOR que la
  regresión (3.17) — supervivencia no es para estimación puntual. Valor real:
  probabilístico/intervalos + usa censura + es el framing del eje. Complementa, no
  reemplaza. Ver `survival_model.py`. Próximo: curvas/intervalos (EXP-A3), Brier.
- ⬜ **EXP-A2 · Predicción dinámica formal (landmarking de supervivencia).** Re-estimar
  la curva de días-restantes en cada día con lo observado hasta ahí. Es el framing
  "canónico" del eje de la tesis.
- ✅ **EXP-A3 · Intervalos.** Cuantiles LightGBM sub-cubren (48%); **CQR (conformal)**
  los calibra a 84% (obj. 80%). Intervalo honesto ~±5 días. `conformal_intervalos.py`.
- ⬜ **EXP-A4 · Clasificación por ventana:** "¿cosecha en los próximos 7/14 días?"
  (probabilístico, quizá más accionable para planificar logística).
- ⬜ **EXP-A5 · Predecir `D` total vs predecir `días_restantes` directo** — comparar si
  cambia algo modelar el residual.

### B. Modelos / algoritmos
- 🟡 **EXP-B1 · Baselines fuertes:** hecho el "prom. por mes causal" (7.06). Faltan
  lineal/KNN si se quiere.
- ✅ **EXP-B2 · Comparar familias + tuning:** LightGBM 2.97 · XGBoost 3.01 · HistGBR 3.10
  (dentro del ruido). Champion final (LGBM+trayectoria+clima+historia) = **2.80**.
- ⬜ **EXP-B3 · Modelos secuenciales:** algo que consuma la serie diaria como secuencia
  (p. ej. features de series temporales tipo `tsfresh`, o un modelo recurrente chico).
  Ojo con la muestra (186) → riesgo de sobreajuste; medir con cuidado.
- ⬜ **EXP-B4 · Tuning honesto** con CV anidada (no tocar el test al elegir hiperparám.).

### C. Feature engineering
- ⬜ **EXP-C1 · Features de tendencia más ricas:** pendientes por ventana (últimos 3/7/14
  días), aceleración, día del último mínimo/máximo de PPM.
- ⬜ **EXP-C2 · Ratios y estados fisiológicos:** días desde entrada a tanque, tiempo en
  pre-tanque, nº de tanques distintos usados, estabilidad del pH.
- ⬜ **EXP-C3 · Densidad/tasa de eventos** por fase del ciclo (ya vimos que la tasa
  cruda no ganó al conteo, pero por fase puede).
- ⬜ **EXP-C4 · Estacionalidad más fina:** día del año continuo, foto-período,
  interacción mes × PPM.
- ⬜ **EXP-C5 · Selección de features / ablations** sistemáticas (grupos: solo pH,
  solo PPM, solo eventos, solo estáticas, combinaciones) → tabla de aportes limpia.

### D. Datos
- ⬜ **EXP-D5 · Pronóstico estacional BACKTESTEABLE (problema abierto).** Ninguna API
  gratis/barata sirve hindcasts estacionales en un punto: Open-Meteo seasonal solo ~1 mes
  atrás; NMME/C3S son gratis pero pesados (grillado/GRIB) y NMME no trae radiación;
  Meteomatics tiene todo pero es enterprise (hay programa académico gratis, aplicar);
  meteoblue tiene estacional 6m + solar y free-trial 10M créditos/año, pero falta
  confirmar si es backtesteable. **Idea de Lucas a explorar**: reconstruir el outlook
  estacional pasado scrapeando pronósticos publicados por un sitio de noticias/clima año
  a año (misma fuente, para consistencia) y derivar un valor mensual esperado. Sería una
  feature de "expectativa pública" backtesteable, aunque grosera. Evaluar factibilidad y
  aporte antes de invertir.
- ⬜ **EXP-B2 (pendiente de cierre) · Benchmark "promedio por mes CAUSAL".** En
  `tabla_final.py` el promedio por mes de inicio debe calcularse SOLO con cosechas que ya
  cerraron ANTES del inicio de la cosecha en curso (ventana temporal expansiva, past-only),
  no con todo el train fold. Corregir y correr la tabla consolidada de los 6.
- ✅ **EXP-D1 · Datos climáticos externos.** Enrichment armado en
  `../dataset_enrichment/` (Open-Meteo, Bariloche −41.1078/−71.1945), **un dataset por
  API** con extractor + README c/u:
  - `era5_real/` — clima real diario 2022-2026 (34 vars: temp, DLI/luz, nubosidad,
    lluvia, humedad, viento + derivadas GDD/amplitud).
  - `pronostico_lead/` — Previous Runs API: pronóstico pasado **con error real**, lead
    1-7 d, cubre todo el dataset. **Backtesteable.** Sanity: MAE temp 0.79°C (lead 1) →
    1.70°C (lead 7), corr 0.99→0.93 (el error crece con el lead = pronóstico genuino).
  - `climatologia/` — normales por día del año (futuro esperado lejano, leakage-free).
  - `pronostico_vivo/` — Forecast API 16 d (producción).
  - `seasonal/` — Seasonal API SEAS5, outlook 3-6 meses ensemble (SOLO producción:
    backtesteable apenas ~1 mes, `past_days` devuelve NaN para el pasado → no sirve para
    training). Descartada la Historical Forecast API (= ERA5, sin error).
  **Sanity confirmado: el clima calca la estacionalidad de la duración** (verano DLI 62
  → ~37d; invierno DLI 10 → ~60d).
  PENDIENTE: cruzar por FECHA y armar features pasado/ventana-60d/futuro(≤7d pronóstico +
  climatología) con diseño anti-leakage, y medir el aporte al MAE. Horizonte estacional
  backtesteable requiere hindcasts NMME/IRI o C3S (evaluar si vale).
- ⬜ **EXP-D2 · Confirmar factor de conversión PPM** (500 vs 640) con la empresa y
  re-evaluar sensibilidad del modelo al supuesto.
- ⬜ **EXP-D3 · Robustez a faltantes:** cómo cambia el MAE si degradamos artificialmente
  la cantidad de mediciones (simular registro más pobre).
- ⬜ **EXP-D4 · Efecto de la censura / cierres parciales** en las conclusiones.

### E. Validación y honestidad metodológica
- ✅ **EXP-E1 · Validación temporal.** Producción (train pasado→test futuro) da ~3.7-4.1 d
  (con train suficiente), PEOR que la agrupada (2.80) → la agrupada es optimista. Reportar
  ambas. `validacion_temporal.py`.
- ⬜ **EXP-E2 · Curva de aprendizaje** (MAE vs nº de cosechas de train) → cuantifica el
  argumento de tamaño de muestra y cuánto ganaríamos con más datos.
- ⬜ **EXP-E3 · Intervalos de confianza por bootstrap** sobre las métricas.
- ⬜ **EXP-E4 · Análisis de residuos:** ¿dónde falla? (cosechas largas, cierta época,
  tandas sin tanque).

### F. Interpretabilidad y factores accionables (arista de la tesis)
- ✅ **EXP-F1 · SHAP** (nativo LightGBM). Accionables pesan poco (1.07 vs 12.23).
  Más PPM→más corto; pH bajo/estable→más corto; Bovel marca ciclos largos (causalidad
  inversa). Caveat causal. `shap_actionables.py`, `shap_accionables.png`.
- ⬜ **EXP-F2 · Dependencia parcial** de las variables controlables por el operador.
- ⬜ **EXP-F3 · Traducir a recomendaciones** accionables y discutir causalidad
  (asociación ≠ causa; qué se podría validar en la práctica).

### G. Otras aristas del problema
- ⬜ **EXP-G1 · Germinación temprana** (predecir al inicio según época + condiciones).
- ⬜ **EXP-G2 · Rendimiento** (exploratoria: ¿se puede estimar con estadísticos de la
  trayectoria?). NOTA: arista secundaria, no eje.

---

## 3. Bitácora de resultados

Se completa a medida que corremos experimentos. Formato sugerido por entrada:
ID · fecha · qué se probó · setup/diff vs baseline · resultado (métrica) · lectura.

| ID | Fecha | Qué | Resultado clave | Lectura |
|---|---|---|---|---|
| EXP-01 | 2026-08-23 | Landmark regression MVP (HistGBM) | MAE 3.20 d (dummy 8.19; estáticas 4.42) | Viable. Trayectoria aporta +27.6 %. |
| EXP-D1 | 2026-08-23 | + enrichment climático (era5+pronóstico7d+climatología), 17 features | MAE 3.07 d (sin clima 3.20) | Clima aporta +4.1 % neto. Pero las features de clima son las MÁS importantes (cl_fc_dli_next7, cl_prev60_temp, cl_fut14_dli_clim) y desplazan a `mes` del top → son el MECANISMO físico de la estacionalidad; el aporte neto es chico por redundancia con las features de fecha. Mejora en etapas temprana/media, empata en la cola. |
| EXP-B2 | 2026-08-23 | Shootout tuneado (RandomizedSearchCV, GroupKFold), all features | LightGBM 2.97 · XGBoost 3.01 · HistGBR 3.10 (baseline sin tunear ~3.15) | Tuning + mejor algo bajan ~6 % (3.15→2.97). Los tres DENTRO DEL RUIDO (±0.5). El modelo no es el cuello de botella, la muestra sí. Best LightGBM: n_est 800, lr 0.03, num_leaves 63. NOTA: XGB/LGBM requieren libomp — se enlazó el de sklearn al lib de uv. best_score_ con leve optimismo → confirmar con CV anidada. |
| tabla_final | 2026-08-24 | Benchmark "prom. por mes CAUSAL" (past-only) + tabla de 6 | Dummy 8.25 · Prom-mes-causal 7.06 (+14.5%) · estáticas 4.40 · completo 3.17 · +clima 3.15 · LightGBM 3.01 | El benchmark por mes causal le gana claro al dummy; pero el salto grande es de los modelos. `tabla_final.py`. |
| EXP-A1 | 2026-08-24 | Supervivencia (Random Survival Forest) con censura | C-index 0.896 · MAE tiempo restante 3.83 (peor que regresión 3.17) | Usa las 62 censuradas. Fuerte en discriminación/probabilístico, no en punto. Complementa. `survival_model.py`. |
| EXP-A3 | 2026-08-24 | Intervalos (regresión por cuantiles LightGBM), tiempo restante | p50 MAE 2.75 · **cobertura [p10,p90] 56% vs 80% objetivo** (sub-cubierto) | Intervalos overconfident → FIX: conformal prediction. Bandas se angostan al avanzar (6.8→4.7d). `intervalos.py`, `int_*.png`. |
| historia | 2026-08-24 | Features causales de duración pasada + análisis por mes | **base+historia MAE 2.89 (vs 3.16 sin)**, +8.5%. `hist_dur_reciente5` = feature #1 | La historia de duraciones es MUY predictiva (prom. últimas 5 cerradas, y mismo mes). Por mes: abr-jul largos/inciertos (MAE 3.2-3.7), ago-sep-dic cortos/predecibles (2.1). `analisis_pasado_mes.py`, `mes_analisis.png`. |
| EXP-D5 | 2026-08-24 | Agente extractor PCT del SMN (texto→terciles) | Prototipo `parse_pct` funciona (2/3 exactos) | Fuente: Pronóstico Climático Trimestral SMN (mensual, terciles temp/precip Patagonia). Backtesteable vía IRI/NMME (estructurado) o parseo de reportes/notas (no estructurado). `dataset_enrichment/pronostico_smn/`. Falta ingesta de reportes pasados + LLM para parseo robusto. |

---

## 4. Preguntas abiertas / ideas sueltas
- ¿Conviene modelar a nivel **tanda** o **tanque** (deduplicando tandas que comparten
  tanque)? Impacto en el peso de casos casi-duplicados.
- ¿El "día actual" como feature mete un sesgo de supervivencia que haya que tratar
  explícitamente en el framing de regresión? (En supervivencia se resuelve solo.)
- ¿Vale unificar unidades de PPM o mejor z-score por `UNIDAD` dentro de cada fold?
- ¿Cuánto del error es irreducible (ruido de medición manual, decisión humana de
  cuándo cosechar)? Definir un "techo" razonable.
