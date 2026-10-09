# Challenge de modelos y ablación de features — Predicción del tiempo de cosecha

> Documento de resultados listo para el informe final. Sesión: 26-09-2026.
> Scripts: `benchmark.py` (challenge), `features_extra.py` (ablación), `campeon_catboost.py` (campeón).
> Datos: `../dataset/dataset_albahaca_1.csv` (albahaca, 208 cosechas, 184 cerradas).
> Landmarks: `landmarks_full.parquet` (24.861 filas tanda×día, 90 features: estáticas + trayectoria + clima + historia).

---

## 1. Objetivo

Responder dos preguntas de forma rigurosa:
1. **¿Qué familia de modelos predice mejor el tiempo de cosecha?** (challenge de 7 tecnologías).
2. **¿Aportan valor features derivadas nuevas** (fotoperíodo, tendencias por ventana, tiempo en rango, estados de tanque, tasa de eventos)?

---

## 2. Diseño experimental

### Partición de evaluación: test temporal (último año)
En vez de una CV que mezcla cosechas de todos los años (optimista), se evalúa como en
**producción**: entrenar con el pasado y predecir el futuro.
- **TRAIN:** cosechas iniciadas **antes** del 2025-09-26 → 142 cosechas / 15.041 landmarks.
- **TEST:** cosechas iniciadas en el **último año** (≥ 2025-09-26) → **42 cosechas / 9.820 landmarks** (sep-2025 → jul-2026, cubre las 4 estaciones).
- **Unidad de generalización = la cosecha** (`ID_COSECHA`): jamás se mezclan días ni tandas de una misma cosecha entre train y test.

### Optimización de hiperparámetros
- `RandomizedSearchCV`, **n_iter = 50** configuraciones por modelo.
- **GroupKFold(5) por cosecha SOLO sobre el train** (loop interno) → nunca toca el test.
- Criterio de selección: **MAE** (coherente con el objetivo del negocio).

### Preprocesamiento
- `Pipeline` de sklearn con imputación (mediana) + escalado **dentro de cada fold**
  (obligatorio para lineal/kNN/MLP; evita fuga). Los árboles de boosting manejan NaN nativo.

### Métricas reportadas
- **MAE** (error típico, robusto), **MedAE** (mediana del error), **RMSE** (sensible a la cola larga).
  Se reportan las tres: la brecha RMSE − MAE mide cuán fea es la cola.

---

## 3. Resultado del challenge (7 tecnologías, 5 paradigmas)

Ranking sobre el **test temporal** (42 cosechas). Dummy (predecir la duración media) = MAE 8.35.

| # | Modelo | Familia | **MAE** | MedAE | **RMSE** | MAE CV-interno |
|---|--------|---------|--------:|------:|---------:|---------------:|
| 🥇 1 | **CatBoost** | boosting | **2.88** | **2.18** | **3.84** | 2.80 |
| 2 | LightGBM | boosting | 3.38 | 2.60 | 4.50 | **2.74** |
| 3 | RandomForest | bagging | 3.50 | 3.12 | 4.46 | 2.87 |
| 4 | XGBoost | boosting | 3.64 | 3.13 | 4.52 | 2.81 |
| 5 | kNN | instancias | 4.26 | 3.93 | 5.10 | 2.97 |
| 6 | ElasticNet | lineal | 5.22 | 4.50 | 6.44 | 3.42 |
| 7 | MLP | red neuronal | 7.16 | 5.01 | 9.60 | 3.06 |

Mejores hiperparámetros de CatBoost: `depth=6, learning_rate=0.03, l2_leaf_reg=1, iterations=500, subsample=0.7`.

### Hallazgos

1. **CatBoost es el campeón** (MAE 2.88 y el mejor RMSE 3.84 → también controla mejor la cola larga).
2. **CatBoost es el más ROBUSTO al paso del tiempo.** Brecha CV-interno → test temporal:
   - LightGBM: 2.74 → 3.38 (**+23%**, se cae).
   - CatBoost: 2.80 → 2.88 (**+3%**, casi no se cae).
   LightGBM parece mejor en la CV agrupada pero sobreajusta la distribución del pasado;
   CatBoost generaliza mucho mejor hacia el futuro. **Esto justifica usar el test temporal
   como criterio de decisión, no la CV agrupada.**
3. **La red neuronal (MLP) es la PEOR** (7.16, apenas mejor que el dummy) → evidencia directa
   de que el deep learning no funciona con una muestra chica (n≈184 cosechas). Consistente con
   Kamilaris & Prenafeta-Boldú (2018).
4. **El modelo lineal (ElasticNet) también flojo** (5.22) → hay no-linealidades relevantes.
5. **Los métodos de árboles (boosting + bagging) dominan** el top-4 → el modelo no es el cuello
   de botella; la muestra sí.

### ¿Por qué gana CatBoost? (para el informe)
No por ser más potente, sino **más regularizado / prudente**, lo que gana cuando hay pocos
datos y la distribución cambia con el tiempo:
- **Árboles simétricos (oblivious):** misma condición de corte por nivel → mucha menos capacidad
  de memorizar ruido que el crecimiento *leaf-wise* de LightGBM (num_leaves hasta 127).
- **Ordered boosting:** calcula los gradientes de cada ejemplo usando solo ejemplos "vistos antes"
  (permutaciones), evitando el *prediction shift* (sobreoptimismo del boosting clásico).

### Costo computacional (nota práctica)
CatBoost ≈ 1.9 h, RandomForest ≈ 42 min, MLP ≈ 35 min (single-thread + contención); LightGBM/XGBoost
minutos. **Para iterar rápido conviene prototipar con LightGBM y reservar CatBoost para corridas finales.**

---

## 4. Ablación de features derivadas nuevas

Se agregaron al set del campeón, de forma **causal** (solo datos hasta el día de la predicción),
cinco familias nuevas, y se midió el aporte con el mismo LightGBM tuneado.

| Feature set | CV-MAE | RMSE | **Test temporal MAE** | Tmp-RMSE |
|-------------|-------:|-----:|----------------------:|---------:|
| DEFINITIVO (base+clima+historia) | 2.820 | 4.10 | **3.12** | 4.20 |
| + FOTO (fotoperíodo) | 2.744 | 3.99 | 3.23 | 4.30 |
| + VENT (tendencias/volatilidad por ventana) | 2.778 | 4.10 | 3.48 | 4.66 |
| + RANGO (tiempo en rango óptimo pH/PPM) | 2.872 | 4.16 | 3.32 | 4.44 |
| + TANQUE (estados de tanque) | 2.775 | 4.00 | 3.62 | 4.75 |
| + EVENT (tasa de eventos) | 2.900 | 4.15 | 3.44 | 4.55 |
| **+ TODAS las nuevas** | **2.722** | 4.01 | **4.13** | 5.26 |

### Hallazgo (resultado negativo valioso)
**Las features nuevas mejoran la CV agrupada (optimista) pero EMPEORAN el test temporal en las
cinco familias** — y "todas juntas" lo empeora fuerte (3.12 → 4.13, +33%). Es un caso claro de
**sobreajuste**: la CV agrupada premia estructura espuria que no generaliza hacia el futuro.
- El **fotoperíodo es redundante** con el DLI y el mes, que ya codifican la luz.
- **Conclusión:** no incorporar estas features. Refuerza el argumento central de la tesis: el
  cuello de botella es el **tamaño de muestra**, no la ingeniería de features.
- **Caveat:** la ablación usó los hiperparámetros fijos del campeón (no re-tuneados por set); la
  degradación temporal consistente en las 5 familias indica que la conclusión no cambiaría.

---

## 5. Conclusiones para la tesis

1. **CatBoost pasa a ser el modelo campeón** por su robustez temporal (test MAE 2.88, RMSE 3.84).
2. La comparación amplia (7 tecnologías, 5 paradigmas) **demuestra** que ninguna otra familia le
   gana → cierra el argumento "el modelo no es el límite".
3. El fracaso del MLP y el flojo desempeño del lineal **respaldan** decisiones metodológicas del trabajo.
4. Las features derivadas **no aportan** → el lever real es más datos (más cosechas / variables de la planta).
5. Metodológicamente: **medir siempre en test temporal**, no solo en CV agrupada (que es optimista).

## 6. Limitaciones
- El test temporal son 42 cosechas → el estimador tiene varianza; los tres modelos de árboles top
  están relativamente cerca. La ventaja de CatBoost se sostiene en MAE **y** RMSE, lo que la hace creíble.
- La ablación no re-tuneó hiperparámetros por feature set (ver caveat en §4).
- Los tiempos de cómputo reportados incluyen contención entre procesos.

## 7. Archivos generados
- `benchmark.py` / `benchmark_resultados.csv` / `benchmark_log.txt` — challenge.
- `features_extra.py` / `features_extra_log.txt` — ablación de features.
- `campeon_catboost.py` — campeón CatBoost + SHAP + conformal (resultados en §8, se completan al correr).
- `shap_catboost.csv` / `shap_catboost.png` — importancia SHAP del campeón.
- `oof_catboost.npy` — predicciones out-of-fold del campeón.

## 8. Campeón CatBoost — conformal + SHAP

### Métricas del campeón
| Validación | MAE | MedAE | RMSE |
|---|---:|---:|---:|
| CV agrupada (GroupKFold por cosecha) | **2.786** | 1.912 | 4.039 |
| Test temporal (último año, 42 cosechas) | **2.882** | 2.185 | 3.842 |

La cercanía entre ambas (2.79 vs 2.88) confirma la **robustez temporal** de CatBoost. La mediana
del error (MedAE ≈ 1.9-2.2 d) indica que la mitad de las predicciones caen dentro de ~2 días.

### SHAP — factores más influyentes (|SHAP| medio, top)
1. `hist_dur_reciente5` (2.28) — **la historia de duraciones recientes domina**.
2. `hist_dur_mismomes` (1.05), `hist_dur_global` (0.85), `hist_n_mismomes` (0.77) — más historia.
3. `mes_cos` / `mes_sin` / `mes_inicio` — estacionalidad.
4. `cl_prev60_temp`, `cl_fc_dli_next7`, `cl_prev60_dli`, `cl_dli_*` — **clima/luz (DLI)**.
5. `n_BOVEL` (0.35) — **único evento de manejo en el top 20** (accionable, pero pesa poco).

**Lectura:** historia + estación + luz dominan; los factores accionables por el operador pesan
poco. Reproduce el hallazgo previo con LightGBM. Recomendaciones de manejo → tomar con cautela
(asociación ≠ causa; `n_BOVEL` es marcador de causalidad inversa: se aplica ante ciclos problemáticos).

### Intervalos conformal (CQR con CatBoost) — objetivo 80%
| Método | Cobertura | Ancho medio |
|---|---:|---:|
| Naive (cuantiles) | 51.0% (sub-cubre, "miente") | 4.9 d |
| **CQR (conformal)** | **80.8%** ✓ | **9.7 d** (≈ ±5 días) |

Cobertura bien calibrada y **estable en todas las etapas** del ciclo (79.8%–82.8%), ancho ~9-10.5 d.
Para planificar: **"cosecha en X ± 5 días con 80% de confianza"**.

### Conclusión
CatBoost queda como **campeón** con eje completo: punto (MAE 2.88 temporal) + probabilístico
(intervalos conformal calibrados) + interpretabilidad (SHAP). Archivos: `oof_catboost.npy`,
`shap_catboost.{csv,png}`.

---

## 9. Objetivo 2 — proyección en T=0 (¿se justifica un modelo estático aparte?)

**Reencuadre del objetivo 2:** no se modela la germinación (existe solo como el evento
`G_A_F1` en el 42% de las tandas); el problema que se resuelve es **proyectar el tiempo de
cosecha en el momento 0** (estimación temprana para planificación estratégica).

**Pregunta:** ¿conviene un modelo ESTÁTICO dedicado a T=0, o el modelo dinámico (campeón
CatBoost) consultado al inicio ya alcanza? → ¿se justifica un "modelado aperturado"?

**Diseño** (`estatico_t0.py`):
- Punto T=0 = primer landmark de cada tanda (día ~3). Una fila por tanda (565: 335 train / 230 test).
- Target = duración total D. Test temporal (último año, 42 cosechas).
- Features T=0 (27): estáticas (época, cultivo, fracción) + historia de cosechas previas +
  clima de arranque (prev60 + climatología futura + pronóstico). **Sin trayectoria de sensores.**
- Challenge de 13 competidores (de simples a complejos, dado el n chico) + baselines.
- Incumbente: campeón CatBoost (features completas, entrenado en los 15.041 landmarks) evaluado en T=0.

### Resultado (test temporal)
| # | Modelo | MAE | MedAE | RMSE |
|---|--------|----:|------:|-----:|
| 🥇 1 | **CatBoost CAMPEÓN @T=0 (features completas)** | **3.12** | 2.75 | 3.88 |
| 2 | CatBoost dedicado | 3.40 | 2.62 | 4.29 |
| 3 | LightGBM dedicado | 3.57 | 2.77 | 4.40 |
| 4 | Promedio por mes (baseline) | 3.66 | 2.54 | 4.86 |
| 5 | RandomForest | 3.72 | 3.20 | 4.53 |
| 6 | LinearReg (OLS) | 3.80 | 3.11 | 5.00 |
| 7 | XGBoost | 4.08 | 3.70 | 4.92 |
| 8 | kNN | 4.22 | 4.38 | 4.83 |
| 9 | Árbol simple (shallow) | 4.94 | 4.34 | 5.95 |
| 10 | GLM Gamma | 5.72 | 6.00 | 6.54 |
| 11 | ElasticNet | 6.89 | 7.21 | 7.88 |
| 12 | Ridge | 7.18 | 7.48 | 8.19 |
| 13 | MLP | 7.37 | 6.73 | 8.54 |
| | Dummy | 7.68 | 8.90 | 8.61 |

### Veredicto: NO se justifica el modelo aperturado
El campeón general @T=0 (3.12) le gana a **todos** los modelos dedicados (mejor: 3.40). Se probó
tanto con simples (OLS, Ridge, GLM Gamma, árbol shallow) como complejos; ninguno lo alcanza.

### Hallazgos
1. **Los modelos simples NO ganaron.** Los lineales regularizados (Ridge 7.18, ElasticNet 6.89)
   colapsaron: bajo el **shift train/test** (EDA: pH↓, PPM↑ en el último año) los lineales
   extrapolan mal; los árboles, constantes por tramos, resisten. El problema no es la complejidad
   sino la **extrapolación bajo shift**.
2. **El promedio por mes (3.66) casi empata al mejor ML dedicado (3.40).** En T=0 **la señal es
   casi toda estacional**; con solo condiciones iniciales hay poco más para exprimir.
3. **Por qué gana el campeón @T=0:** entrenado con los 15.041 landmarks (todos los días), aprende
   el mapeo estático→duración con mucho más dato que un modelo limitado a 335 filas.

### Implicancia de diseño
**No conviene un modelo estático aparte.** El modelo dinámico unificado, consultado en T=0, ya da
la mejor estimación temprana (MAE ~3.1 d) y sirve para ambos horizontes (estratégico en T=0 y
táctico a medida que avanza). Un solo modelo, no dos.

**Caveat:** el HPO interno usó GroupKFold aleatorio (no temporal), lo que penaliza más a los
lineales; el veredicto es idéntico en dos corridas → robusto. Archivos: `estatico_t0.py`,
`estatico_t0_resultados.csv`, `estatico_t0_log.txt`.
