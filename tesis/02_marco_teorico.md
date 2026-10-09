# Capítulo 2 — Marco teórico y estado del arte

## 2.1 Aprendizaje automático para la predicción en agricultura

La predicción de resultados productivos con datos es un campo consolidado. La revisión
sistemática de van Klompenburg et al. (2020) relevó decenas de estudios de predicción de
rendimiento con aprendizaje automático e identificó que los algoritmos más usados son las redes
neuronales, los bosques aleatorios y los métodos de *boosting*, aplicados sobre variables
ambientales y de manejo, siendo la temperatura y la precipitación las más influyentes. Liakos et
al. (2018) sistematizan el uso del aprendizaje automático en agricultura como herramienta de
apoyo a la decisión, y Kamilaris y Prenafeta-Boldú (2018) documentan el crecimiento del
aprendizaje profundo en el dominio, señalando su buen desempeño pero también su **dependencia de
grandes volúmenes de datos** —una advertencia central para este trabajo, cuya muestra es
acotada. En una línea complementaria, Wolfert et al. (2017) y Kamilaris et al. (2017) describen
el estado y los desafíos del *big data* en la agricultura, destacando que gran parte del dato
generado permanece subutilizado.

## 2.2 Agricultura de ambiente controlado e hidroponía

La hidroponía y las *plantas fábrica* se conciben como sistemas de producción altamente
instrumentados y optimizables (Kozai et al., 2015). Sambo et al. (2019) revisan la gestión de la
solución nutritiva en sistemas sin suelo, enmarcando el tipo de variables (pH, conductividad)
con las que se trabaja; Benke y Tomkins (2017) sitúan la agricultura de ambiente controlado como
una respuesta sustentable frente a la presión sobre los recursos. En el cultivo específico,
Saha et al. (2016) estudian el crecimiento y rendimiento de la albahaca (*Ocimum basilicum* L.)
en sistemas sin suelo. En el terreno aplicado del aprendizaje automático, Mokhtar et al. (2022)
compararon varios modelos para predecir el rendimiento de lechuga hidropónica con datos del
propio sistema, obteniendo errores acotados, y Mamatha y Kavitha (2023) integraron sensores y
modelos para la gestión del crecimiento en invernaderos hidropónicos.

## 2.3 Análisis de tiempo al evento y predicción dinámica

El problema de "cuánto falta hasta que ocurra un evento" tiene una tradición sólida en la
estadística y la medicina. El estimador de Kaplan y Meier (1958) y el modelo de riesgos
proporcionales de Cox (1972) son la base del análisis de supervivencia, que maneja de forma
natural la **censura** (casos aún no ocurridos). Ishwaran et al. (2008) extendieron estas ideas
al aprendizaje automático con los *random survival forests*, y Harrell et al. (1996)
formalizaron el **índice de concordancia (C-index)** como medida de discriminación.

La predicción **dinámica** —re-estimar el tiempo al evento a medida que se acumula información—
se formaliza mediante el *landmarking* (van Houwelingen, 2007; van Houwelingen & Putter, 2011) y
los modelos conjuntos (Rizopoulos, 2011). Esta misma lógica de tiempo al evento se aplica en
ingeniería a la estimación de la **vida útil remanente** (RUL) de maquinaria (Si et al., 2011;
Lei et al., 2018). El presente trabajo adapta ese instrumental maduro —hasta donde se relevó,
por primera vez— a la estimación del tiempo de cosecha en un cultivo hidropónico real.

## 2.4 Cuantificación de la incertidumbre: predicción conforme

Para que una predicción sea útil en decisiones que cuestan dinero, no alcanza con un valor
puntual: hace falta un **intervalo con garantía de cobertura**. La predicción conforme provee
ese marco sin supuestos distribucionales fuertes (Vovk et al., 2005; Shafer & Vovk, 2008;
Angelopoulos & Bates, 2023). Lei et al. (2018) desarrollan la inferencia predictiva libre de
distribución para regresión, y Romano et al. (2019) proponen la **regresión cuantílica
conformalizada (CQR)**, que produce intervalos adaptativos a la heterocedasticidad —el método
que este trabajo emplea para calibrar sus intervalos.

## 2.5 Modelos de aprendizaje automático tabular

Los métodos basados en árboles dominan los problemas tabulares. Breiman (2001) introdujo los
*random forests* (bagging), y Friedman (2001) el *gradient boosting*. Sus implementaciones
modernas —XGBoost (Chen & Guestrin, 2016), LightGBM (Ke et al., 2017) y CatBoost— ofrecen alto
desempeño, manejo nativo de valores faltantes y distintas estrategias de regularización. CatBoost
se distingue por el *ordered boosting* y los árboles simétricos, que reducen el sobreajuste,
característica relevante cuando la muestra es chica. La interpretabilidad de estos modelos se
aborda con los valores SHAP (Lundberg & Lee, 2017; Molnar, 2022), que permiten pasar de predecir
a explicar y, eventualmente, a recomendar.

## 2.6 Validación y la escala de la muestra

Un eje metodológico central es que la **validación debe respetar la estructura de los datos**.
Roberts et al. (2017) muestran que, con datos de estructura temporal o jerárquica, una validación
cruzada que ignore esa estructura subestima el error y produce resultados demasiado optimistas;
la solución es validar por bloques/grupos. Kohavi (1995) estableció las buenas prácticas de
validación cruzada, y Cawley y Talbot (2010) advierten sobre el sesgo de selección: optimizar
hiperparámetros y reportar sobre la misma validación infla el desempeño, lo que motiva la
validación cruzada anidada. Bergstra y Bengio (2012) mostraron que la búsqueda aleatoria de
hiperparámetros es competitiva frente a la búsqueda en grilla. Estas ideas son el fundamento del
diseño de evaluación de este trabajo, que valida agrupando por cosecha y, además, temporalmente.

## 2.7 Trabajos relacionados y el vacío que llena esta tesis

Varios trabajos abordan el problema central o problemas vecinos. El más directo es el de Liu et
al. (2022), que predice el tiempo de cosecha combinando selección de variables y redes de memoria
de largo plazo sobre datos de producción y clima, mostrando que la precisión mejora al acercarse
el evento. Mokhtar et al. (2022) y Mamatha y Kavitha (2023) trabajan el rendimiento y la gestión
en hidroponía; Gómez Arango (2024) aplica aprendizaje automático a la predicción de rendimiento
de cultivos en el ámbito regional.

La lectura conjunta revela un **patrón y un vacío**: la mayoría de los trabajos predice el
rendimiento (o el tiempo) como una magnitud **estática al final del ciclo**, y valida con
particiones que no siempre respetan la estructura de los datos. Solo algunos se aproximan a la
re-estimación temporal, y ninguno —en el relevamiento realizado— adopta el encuadre de **tiempo
al evento por cosecha con validación agrupada y temporal**, ni cuantifica la incertidumbre con
predicción conforme en este dominio. Ese es el espacio donde esta tesis busca aportar: no un
método nuevo, sino la **adaptación rigurosa de instrumental consolidado a un problema productivo
real, con la escala de la muestra como protagonista**.
