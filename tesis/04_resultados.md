# Capítulo 4 — Resultados

## 4.1 Eje: predicción del tiempo de cosecha

### 4.1.1 Viabilidad y progresión
Partiendo de un predictor trivial (la duración media, MAE 8.19 d), el desempeño mejora de forma
sostenida al incorporar información: un modelo con solo variables de contexto (época, cultivo,
día del ciclo) alcanza MAE 4.42 d (−46 %), y al sumar la **trayectoria** de sensores y eventos
baja a 3.20 d (−61 % respecto del trivial). La trayectoria aporta un **+27.6 %** sobre el modelo
de solo contexto, y su aporte **crece a medida que avanza el ciclo**. El modelo refinado
(*boosting* ajustado + trayectoria + clima + historia de duraciones) llega a **MAE 2.80 d**
(MedAE 1.95) en validación cruzada agrupada por cosecha. La *historia* de duraciones recientes
resultó el predictor individual más fuerte.

### 4.1.2 Comparación de tecnologías (challenge)
Se compararon siete tecnologías de cinco familias, con optimización de hiperparámetros y
evaluación en **test temporal** (último año, 42 cosechas). Resultados (duración, días):

| Modelo (familia) | MAE | MedAE | RMSE |
|---|---:|---:|---:|
| **CatBoost** (boosting) | **2.88** | 2.18 | 3.84 |
| LightGBM (boosting) | 3.38 | 2.60 | 4.50 |
| RandomForest (bagging) | 3.50 | 3.12 | 4.46 |
| XGBoost (boosting) | 3.64 | 3.13 | 4.52 |
| kNN (instancias) | 4.26 | 3.93 | 5.10 |
| ElasticNet (lineal) | 5.22 | 4.50 | 6.44 |
| MLP (red neuronal) | 7.16 | 5.01 | 9.60 |
| *Dummy* | 8.35 | 7.60 | 9.18 |

**CatBoost resultó campeón**, con el mejor MAE y el mejor RMSE (mejor control de la cola larga).
El desempeño pobre del MLP (apenas mejor que el trivial) y del modelo lineal confirma que, con
esta escala de muestra, el aprendizaje profundo no es adecuado y que existen no-linealidades que
un modelo lineal no captura.

### 4.1.3 Validación temporal vs. agrupada
La validación cruzada agrupada es **optimista**: la validación temporal (producción real) eleva
el error a ~3.7–4.1 d para la mayoría de los modelos. El hallazgo más relevante es la
**robustez temporal diferencial**: LightGBM cae de 2.74 (agrupada) a 3.38 (temporal, +23 %),
mientras CatBoost pasa de 2.80 a 2.88 (apenas +3 %). CatBoost, por sus árboles simétricos y su
*ordered boosting* (más regularizado), **generaliza mucho mejor hacia el futuro**, lo que
justifica adoptarlo como campeón y usar el test temporal como criterio de decisión.

### 4.1.4 Cuantificación de la incertidumbre (intervalos conformal)
Los intervalos por cuantiles crudos sub-cubren (cobertura empírica 51 % para un objetivo del
80 %). La **regresión cuantílica conformalizada (CQR)** los calibra a **80.8 %** de cobertura,
con un ancho honesto de ~9.7 días (≈ ±5 días), **estable en todas las etapas del ciclo**
(79.8–82.8 %). Esto habilita comprometer fechas con riesgo explícito: *"cosecha en X ± 5 días,
con 80 % de confianza"*.

### 4.1.5 Factores accionables (objetivo 3 de los objetivos específicos)
El análisis SHAP sobre el campeón muestra que la predicción está dominada por la **historia de
duraciones, la estación y la luz (DLI)**; las variables controlables por el operador (pH, PPM,
eventos) pesan poco. La **dependencia parcial (PDP)** lo cuantifica: mover cada variable
accionable de su percentil 5 al 95 cambia la duración predicha en **menos de 0.1 día**, salvo el
conteo de aplicaciones de Bovel (+0.8 d), que es un **marcador de causalidad inversa** (se aplica
ante ciclos problemáticos, no los causa). Conclusión: con los datos disponibles, **el manejo fino
tiene un efecto marginal despreciable sobre la duración del ciclo**, que está gobernada por
factores no controlables (estación/luz). Las recomendaciones de intervención, por tanto, deben
tomarse con cautela y validarse experimentalmente (asociación ≠ causa).

## 4.2 Proyección en el momento 0 (¿modelo estático aparte?)

Para la planificación estratégica interesa estimar la duración **al inicio del ciclo**. Se evaluó
si conviene un modelo estático dedicado a T=0 (con solo información de arranque, sin trayectoria)
frente a consultar el campeón general en ese punto. Compitieron 13 enfoques (de simples a
complejos). **Ninguno superó al campeón general consultado en T=0 (MAE 3.12 d);** el mejor modelo
dedicado alcanzó 3.40 d. Más aún, el **promedio por mes** (3.66 d) casi igualó a los modelos
dedicados, lo que indica que **en T=0 la señal es casi enteramente estacional**. Los modelos
lineales colapsaron (MAE 6.9–7.2) por extrapolación bajo el *shift* temporal. **Conclusión: no se
justifica un modelo estático aparte**; el modelo dinámico unificado, consultado al inicio, ya
provee la mejor estimación temprana y cubre ambos horizontes (estratégico y táctico).

## 4.3 Rendimiento por cosecha (exploratorio)

Sobre el rendimiento atribuido (66 cosechas aisladas, media 1.346 plantas), se compararon
baselines, modelado directo, estimación a partir de la duración y *forecasting*. **El mejor
predictor fue el promedio estacional por mes (MAE ≈ 290 plantas, ~22 %)**; ningún modelo de
aprendizaje automático lo superó. La relación tiempo↔cantidad existe (rendimiento vs duración
ρ ≈ −0.37) pero es **redundante con la estación**, por lo que estimar el rendimiento como función
de la duración predicha no mejora al baseline estacional. Los modelos complejos sobreajustaron
(n = 66), y el *forecasting* de volumen resultó inviable con rigor (solo 20 meses de historia,
MAPE 31 %). **Conclusión exploratoria:** el rendimiento es estimable a ~22 % de error mediante un
promedio estacional, pero los datos actuales —tentativos y escasos— no justifican un modelo
dedicado; se requiere más historia y un registro de rendimiento por cosecha en origen.

## 4.4 Resultados negativos (evidencia a favor de la tesis)

Varios enfoques más sofisticados **no** mejoraron, y cada uno refuerza el mensaje central:
- **El tiempo térmico (GDD) no gobierna el ciclo** (coeficiente de variación 54–110 %; baseline
  agronómico MAE 40 d): en invernadero la temperatura se amortigua y manda la luz (DLI). Justifica
  el enfoque de aprendizaje automático sobre señales más ricas.
- **Las features derivadas no aportan:** fotoperíodo, tendencias por ventana, tiempo en rango,
  estados de tanque y tasa de eventos mejoran la validación agrupada (optimista) pero **empeoran**
  la temporal en todas las familias —sobreajuste bajo *shift*.
- **El forecasting y el modelo estático aparte no se justifican** (secciones 4.2 y 4.3).
- **El aprendizaje profundo (MLP) es el peor** de los modelos probados.

## 4.5 El tamaño de muestra como límite (curva de aprendizaje)

La evidencia más directa del argumento central se obtiene variando el número de cosechas de
entrenamiento y midiendo el MAE en un test temporal fijo:

| Nº de cosechas de entrenamiento | MAE test (días) |
|---:|---:|
| 20 | 4.80 ± 0.93 |
| 40 | 4.11 ± 0.65 |
| 60 | 3.48 ± 0.38 |
| 80 | 3.43 ± 0.36 |
| 100 | 3.32 ± 0.16 |
| 142 (todas) | 3.13 |

El error decrece de forma sostenida (4.80 → 3.13 d) y la curva **aún no se ha aplanado** al usar
todas las cosechas disponibles (pendiente final ≈ −0.01 d/cosecha). Esto demuestra que **el cuello
de botella del desempeño es la cantidad de cosechas, no el modelo**: acumular más ciclos seguiría
reduciendo el error. Es el resultado que articula toda la tesis y el que convierte sus "resultados
negativos" en evidencia coherente.
