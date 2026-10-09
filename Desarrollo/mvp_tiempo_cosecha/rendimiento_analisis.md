# Objetivo 3 — Predicción de la cantidad de plantines (rendimiento)

> Documento de análisis, listo para el informe final. Sesión: 27-09-2026.
> Scripts: `rendimiento.py` (atribución), `rendimiento_modelado.py` (comparación de enfoques).
> Dato nuevo: `../dataset/HGP - Albahacas por cosecha.xlsx` (planilla de negocio).

---

## 1. Contexto y dato

El objetivo 3 busca estimar el **rendimiento** (cantidad de plantas por cosecha). La fuente es
una **planilla de negocio** con la cantidad de albahacas cosechadas **por día calendario**
(2025-2026), NO por cosecha. Total: **104.532 plantas** en 296 días de cosecha.

**No hay vínculo 1:1** con el dataset principal (es otra planilla). Además, algunos días de
valor bajo corresponden a **pedidos puntuales**, no a cosechas reales.

### Atribución tentativa rendimiento → cosecha (`rendimiento.py`)
Cada día de rendimiento se asigna a la cosecha cuyo **cierre (`COSECHA_FIN`) está más cercano
en el tiempo**. No se excluye ningún día (los pedidos chicos se suman al cierre más cercano),
de modo que **se conserva la masa total** de plantas. Se marca cada cosecha como *aislada*
(ningún otro cierre a ±2 días) para tener un subconjunto de etiquetas limpias.

**Supuesto explícito (tentativo):** el volumen del entorno temporal pertenece a la cosecha que
cierra más cerca. Es una aproximación; la planilla no permite una atribución exacta.

**Sanity check:** masa conservada (104.532 = 104.532); distancia media día→cierre = 2.2 d
(máx 10); 82 de 86 cosechas de 2025-26 con rendimiento > 0.

**Limitación dura:** solo **~86 cosechas** etiquetables (2025-26; la planilla no cubre 2022-24),
**66 aisladas** (etiqueta limpia). Muestra chica → análisis **exploratorio**, no confirmatorio.

---

## 2. Marco teórico — enfoques de predicción de rendimiento en casos similares

La predicción de rendimiento agrícola con datos es un campo consolidado. La revisión
sistemática de van Klompenburg et al. (2020) muestra que los enfoques dominantes son de
**aprendizaje automático supervisado** (bosques aleatorios, boosting, redes neuronales) sobre
variables ambientales y de manejo. En agricultura de ambiente controlado, Mokhtar et al. (2022)
predijeron el rendimiento de lechuga hidropónica con errores acotados usando datos del propio
sistema. Se distinguen cuatro familias de enfoque, todas evaluadas en este trabajo:

1. **Modelado directo (regresión supervisada).** Un modelo aprende rendimiento = f(features de
   la cosecha: estación, condiciones iniciales, clima). Es el enfoque estándar de la literatura.
   Ventaja: captura interacciones. Riesgo con muestra chica: sobreajuste.

2. **Estimación mecanística/matemática a partir de otra variable predicha.** Si existe una
   relación estable entre una magnitud ya modelada (aquí, la **duración del ciclo**, que predice
   CatBoost) y el rendimiento, se puede derivar el rendimiento con una **fórmula** (lineal o de
   potencia) en lugar de un modelo aparte. Es el principio de los modelos de crecimiento
   agronómicos (p. ej. relación rendimiento–tiempo térmico). Ventaja: parsimonia, interpretable,
   robusto con pocos datos. Requiere que la relación exista y sea estable.

3. **Modelo combinado.** Usar la **duración estimada por CatBoost como feature** dentro del
   modelo de rendimiento, junto con estación y clima. Combina lo mejor de (1) y (2).

4. **Forecasting de volumen (serie temporal).** En vez de por cosecha, predecir el **volumen
   agregado por período** (semana/mes) para planificación comercial, con métodos estacionales
   (naive estacional, Holt-Winters/ETS, SARIMA). Es lo adecuado cuando la unidad de decisión es
   el abastecimiento agregado y no la cosecha individual.

**Criterio de decisión:** con muestra chica y un target tentativo, la **parsimonia** es clave
(Cawley & Talbot, 2010): si una fórmula sobre la duración de CatBoost iguala a un modelo
dedicado, conviene la fórmula (menos varianza, más interpretable, no requiere mantener otro
modelo). Se compara todo contra baselines honestos (media, promedio por mes).

---

## 3. EDA del rendimiento (`rendimiento.py`)

- **Distribución:** media 1.215 plantas/cosecha, mediana 1.194, sd 618, rango 0–2.784
  (aisladas: media 1.346, sd 475 — más concentrada).
- **Relación tiempo ↔ cantidad (clave):** rendimiento vs duración **Spearman −0.30** → los
  ciclos más largos rinden **menos** plantas. Confirma que existe una relación explotable entre
  la salida de CatBoost (duración) y el rendimiento.
- **Estacionalidad marcada:** rendimiento mediano por mes de cierre va de ~670–770 (jul-ago,
  invierno) a ~1.500–2.000 (mar, oct-dic, primavera/verano). Mismo driver de luz/estación que
  gobierna la duración.
- **Implicación:** estación y duración son los ejes predictivos esperables; el clima de luz
  (DLI) es el mecanismo físico detrás de ambos.
  Figura: `rendimiento_eda.png`.

---

## 4. Comparación de enfoques (`rendimiento_modelado.py`)

Target = rendimiento (plantas), cosechas aisladas 2025-26 (**n=66**, media 1.346 plantas).
Validación KFold(5) para los modelos ML (predicción out-of-fold).

**Correlaciones (Spearman) con el rendimiento:** `cl_fut30_dli` +0.46 · `mes_cos` +0.44 ·
`cl_dli_ciclo` +0.42 · `n_tandas` +0.40 · **`dur` −0.38 · `dur_cb_t0` (CatBoost) −0.36** ·
`cl_prev60_dli` +0.30. → La **estación/luz y el nº de tandas pesan más que la duración.**

### Ranking (MAE en plantas; media real 1.346)
| Enfoque | Features | MAE | MedAE | RMSE |
|---|---|----:|------:|-----:|
| 🥇 **Baseline: promedio por mes** | — | **290** | 213 | 382 |
| ML: LightGBM | T0 | 302 | 216 | 409 |
| ML: kNN | T0+DUR_cb | 302 | 250 | 387 |
| ML: OLS | T0 | 304 | 261 | 392 |
| ML: GLM Gamma | FULL_ciclo | 307 | 246 | 402 |
| … (resto de ML) | varios | 308–327 | | |
| MAT: potencia ~ dur_CatBoost | dur_cb_t0 | 357 | 300 | 463 |
| MAT: potencia ~ dur_real | dur | 358 | 321 | 465 |
| Baseline: media global | — | 376 | 320 | 472 |
| CatBoost / RandomForest (varios sets) | | 311–327 | | (sobreajustan) |

**Forecasting de volumen mensual:** serie de solo **20 meses** (<2 ciclos anuales). Naive
estacional (mismo mes del año previo): **MAE 1.421 plantas/mes, MAPE 31%**. Holt-Winters no
ajustable (requiere ≥2 ciclos completos). El forecasting está **limitado por falta de historia**.

## 5. Conclusiones y recomendación

1. **El mejor predictor del rendimiento por cosecha es el promedio estacional (por mes):
   MAE ~290 plantas (≈22% del promedio).** Ningún modelo ML lo supera; el mejor (LightGBM/OLS
   sobre features T=0) apenas lo iguala (~302).
2. **La relación tiempo↔cantidad existe (dur ρ≈−0.37) pero es redundante con la estación**
   (ambas gobernadas por la luz). Por eso **estimar el rendimiento como fórmula sobre la
   duración de CatBoost (MAT ~357) es PEOR que el promedio estacional**, y agregar la duración
   de CatBoost como feature (`T0+DUR_cb`) no aporta sobre la estación.
3. **Los modelos complejos (CatBoost, RandomForest) sobreajustan** con n=66 y quedan por debajo
   de los simples y del baseline — coherente con el resto de la tesis (la muestra manda).
4. **El forecasting es inviable con rigor** hoy (solo 20 meses de planilla).

**Recomendación (objetivo 3 = exploratorio):** con el dato actual —tentativo y chico— el
rendimiento es **estimable a ~22% de error usando el promedio estacional**, y **no hay evidencia
de que un modelo dedicado ni una fórmula sobre CatBoost aporten valor.** Se reporta como
resultado exploratorio honesto: *"se puede dar una estimación estacional útil; un modelado
sofisticado no se justifica con los datos disponibles; se necesita más historia y una
atribución rendimiento↔cosecha más precisa (idealmente registro por cosecha en origen)."*

**Caveat metodológico:** el baseline "promedio por mes" se calcula in-sample (levemente
optimista); una versión validada quedaría un pelo peor, empatando con el mejor ML — lo que no
cambia la conclusión (ningún enfoque supera a una estimación estacional simple).

### Archivos
`rendimiento.py`, `rendimiento_por_cosecha.csv`, `rendimiento_eda.png`,
`rendimiento_modelado.py`, `rendimiento_modelado_resultados.csv`, `rendimiento_modelado_log.txt`.
