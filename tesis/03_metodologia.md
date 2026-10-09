# Capítulo 3 — Materiales y métodos

## 3.1 Tipo de estudio y caso

El trabajo es un **estudio cuantitativo, observacional y retrospectivo** de modelado
predictivo: no hay intervención ni manipulación experimental de las condiciones de cultivo; se
analiza el registro histórico de una operación en funcionamiento. Corresponde a un **estudio de
caso único** —una operación hidropónica comercial de albahaca situada en Bariloche (Río Negro,
Argentina)—, con foco en la construcción y evaluación de modelos predictivos supervisados y un
componente de análisis de tiempo al evento. Por acuerdo de confidencialidad, la empresa se
mantiene anonimizada. El período cubierto va de julio de 2022 a septiembre de 2026.

## 3.2 Los datos

### 3.2.1 Origen y recuperación
El registro operativo se realiza de forma **manual** en una planilla de cálculo. Una
particularidad del caso es que **parte de la información no está en el texto de las celdas sino
en su formato**: el color de relleno codifica el tanque de conexión y el color del borde
inferior codifica el evento de manejo del día. Un **extractor desarrollado a medida** recupera
esa información —alrededor de 6.600 eventos de manejo— que un exportado a texto plano
descartaría, y produce un conjunto tabular limpio.

### 3.2.2 Unidad de análisis y de generalización
Es central la distinción entre dos escalas:
- **Unidad de análisis:** la *tanda* `(ID_COSECHA, ID_TANDA)`, que sigue su trayectoria diaria
  hasta el evento de cosecha (`COSECHA_FIN`).
- **Unidad de generalización:** la *cosecha* (`ID_COSECHA`), el verdadero número de casos
  independientes. El dataset comprende 211 cosechas y ~29.300 registros diarios; para la
  evaluación se consideran las **≈184 cosechas con al menos una tanda completa**. *Ese es el n
  real, no los miles de filas diarias* — un punto metodológico central del trabajo.

### 3.2.3 Variables
- **De contexto (estáticas):** mes de inicio codificado de forma cíclica, cultivo/fracción, día
  actual del ciclo.
- **De trayectoria (diarias):** pH y PPM (conductividad de la solución nutritiva) resumidos por
  último valor, media, desvío, mínimo, máximo, pendiente y media móvil de 7 días; estado de
  tanque; y eventos de manejo (traspaso/cambio de tanque, agregado de sales, etc.) por conteo y
  recencia.
- **De enriquecimiento climático (externas):** reanálisis diario ERA5, climatología por día del
  año y pronóstico, ancladas a la ubicación de la instalación (prev. 60 días, climatología futura
  y pronóstico de corto plazo son conocidos al inicio y por ende libres de fuga).
- **De resultado (objetivo):** la duración total del ciclo `D` (equivalente a los días restantes,
  pues el día actual se conoce con exactitud); y, para la arista de rendimiento, la cantidad de
  plantas cosechadas por cosecha (ver 3.2.4).

### 3.2.4 Datos de rendimiento
La cantidad de plantas se obtiene de una planilla de negocio con el total cosechado **por día
calendario** (2025–2026), sin vínculo 1:1 con las cosechas. Se atribuye de forma tentativa a cada
cosecha el volumen de los días con cierre más cercano, conservando la masa total. Por ser
tentativo y de muestra chica, el rendimiento se trata como objetivo **exploratorio** (ver 4.3).

## 3.3 Análisis exploratorio (EDA)

Se caracterizó la **distribución del target** (duración: media 47 d, mediana 46, asimetría
positiva +0.53, con cola derecha), la **consistencia temporal** (drift de duración por año,
mejora de la completitud de datos en el tiempo, y el cambio de instrumentación entre hojas —ppm
vs mS/cm), la **consistencia entre entrenamiento y test** (se detectó un *shift* significativo:
test KS p<0.001, con pH menor y PPM mayor en el último año) y la **relación de las variables con
el target** (la luz/DLI es lo más correlacionado con la duración). El EDA guía decisiones
posteriores: la elección de MAE/mediana (por la cola), la validación temporal (por el shift) y la
lectura de la importancia de variables.

## 3.4 Diseño metodológico

### 3.4.1 Encuadre por landmarking
El tiempo de cosecha se modela como un problema de **tiempo al evento con re-estimación diaria**.
Mediante *landmarking*, se genera un ejemplo por cada combinación *(tanda × día observado)*,
usando exclusivamente información disponible hasta ese día para evitar fuga de futuro; el modelo
se re-consulta cada día y refina su estimación a medida que avanza el ciclo.

### 3.4.2 Validación
- **Validación cruzada agrupada por cosecha** (`GroupKFold`): nunca se mezclan días ni tandas de
  una misma cosecha entre entrenamiento y prueba. Da una estimación estable pero optimista.
- **Validación temporal** (entrenar con las cosechas más antiguas, predecir las más recientes):
  refleja el uso en producción y es más exigente por el *shift* temporal. **Se reportan ambas.**
- **Selección de hiperparámetros:** búsqueda aleatoria con validación interna agrupada; se
  discute el sesgo de selección (Cawley & Talbot, 2010).

### 3.4.3 Métricas
Se reportan **MAE** (error típico, robusto a la cola), **MedAE** (mediana del error) y **RMSE**
(sensible a los errores grandes). La brecha RMSE−MAE informa sobre la severidad de la cola. Para
el componente de supervivencia se usan el **C-index** y el *Brier score*; para los intervalos, la
**cobertura empírica**.

## 3.5 Modelos, optimización e interpretabilidad

Se comparó un amplio conjunto de tecnologías de distintas familias (lineales, instancias,
bagging, *boosting* y redes neuronales), con optimización de hiperparámetros por búsqueda
aleatoria y preprocesamiento (imputación y escalado) dentro de cada partición para evitar fuga.
La incertidumbre se cuantifica con **predicción conforme (CQR)**, que entrega intervalos con
cobertura calibrada; la interpretabilidad se aborda con valores **SHAP** y **dependencia parcial
(PDP)**, midiendo el aporte por grupos de variables correlacionadas.

## 3.6 Consideraciones éticas

Los datos corresponden a una operación comercial real y se utilizan con autorización explícita de
su propietario; no contienen datos personales ni información sensible de individuos. La empresa se
presenta anonimizada. El trabajo asume además la **honestidad metodológica** como principio
ético: reconocer la limitación de la escala de muestra, validar de manera rigurosa para no
informar resultados engañosamente optimistas y reportar la incertidumbre, evitando sobrevender
los resultados.
