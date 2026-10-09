# Capítulo 5 — Discusión

## 5.1 Interpretación: el tamaño de muestra como límite

El conjunto de resultados converge en una lectura unificada: **con esta operación, el desempeño
está limitado por la escala de la muestra —el número de cosechas independientes— y no por la
elección del algoritmo.** Tres líneas de evidencia lo sostienen. Primero, la **curva de
aprendizaje** (§4.5): el error decrece de forma sostenida con más cosechas y no se aplana,
indicando que acumular ciclos seguiría mejorando la predicción. Segundo, la **comparación de
tecnologías** (§4.1.2): los cuatro métodos de árboles quedan relativamente juntos, mientras el
aprendizaje profundo —ávido de datos— es el peor; no hay un "modelo mágico" por descubrir.
Tercero, los **resultados negativos** (§4.4): agregar features derivadas, construir un modelo
estático aparte o intentar *forecasting* no mejora, porque el problema no es la falta de señal
sino la cantidad de casos para aprenderla y validarla sin sobreajuste.

Esta interpretación es, además, **prescriptiva**: la inversión de mayor retorno para la operación
no es un modelo más complejo, sino **seguir registrando cosechas** (y, para el rendimiento,
registrarlo por cosecha en origen).

## 5.2 El valor del encuadre dinámico y de la incertidumbre

El encuadre de **tiempo al evento con re-estimación por landmarking** aporta valor concreto: la
estimación arranca gruesa (dominada por la estación) y se afina a medida que la trayectoria se
acumula, acompañando las dos decisiones del negocio —comprometer entregas al inicio
(estratégico) y ajustar la logística sobre la marcha (táctico)—. El análisis de T=0 (§4.2)
mostró que un **único modelo dinámico** cubre ambos horizontes, simplificando el despliegue.

La **cuantificación de la incertidumbre** (§4.1.4) es tan importante como el punto: un intervalo
calibrado de ±5 días al 80 % permite planificar con un margen de riesgo conocido, en lugar de
confiar en una estimación puntual sin medida de confiabilidad. Esto es lo que vuelve las
predicciones efectivamente utilizables en decisiones que cuestan dinero.

## 5.3 Factores accionables y causalidad

El hallazgo de que las variables controlables tienen un efecto marginal despreciable sobre la
duración (§4.1.5) debe leerse con cuidado. No implica que el manejo sea irrelevante para la
producción en general, sino que, **dentro del rango observado y para esta variable-objetivo (la
duración del ciclo), la señal accionable es pequeña frente a los factores no controlables**
(estación, luz, historia). Además, las relaciones identificadas son **asociaciones, no
relaciones causales**: el caso del Bovel —asociado a ciclos largos por aplicarse ante problemas—
ilustra el riesgo de confundir marcador con palanca. Cualquier recomendación de intervención
debería validarse experimentalmente antes de adoptarse.

## 5.4 Validez y limitaciones

- **Escala de la muestra:** ≈184 cosechas (y solo ≈66 con rendimiento atribuible) limitan la
  complejidad de modelos razonable y la precisión de las estimaciones. Es la limitación de fondo,
  explicitada y cuantificada en lugar de ocultada.
- **Una sola operación:** las conclusiones quedan atadas a este cultivo e instalación; su
  traslado a otras requiere cautela.
- **Shift temporal:** el EDA detectó un cambio de condiciones entre entrenamiento y el último año
  (pH y PPM), lo que explica por qué la validación temporal es más exigente y por qué los modelos
  robustos (CatBoost) se imponen.
- **Datos históricos reales:** datos faltantes y cambios de instrumentación (p. ej. la diferencia
  de unidades entre hojas) introducen ruido difícil de eliminar por completo.
- **Rendimiento tentativo:** la atribución del volumen diario a cosechas es aproximada; el
  resultado es exploratorio.
- **Ausencia de variables de la planta:** no se registran mediciones morfológicas (número de
  hojas, cobertura), de alto poder predictivo según la literatura; incorporarlas es una de las
  extensiones de mayor potencial.

## 5.5 Transferencia a otros dominios

El núcleo metodológico es transferible: modelar un **proceso que transcurre en el tiempo y
culmina en un evento**, re-estimándolo a medida que se acumula información, y validando con
respeto a la estructura de los datos. Ese esquema es el mismo de la **estimación de vida útil
remanente (RUL)** en el mantenimiento predictivo industrial (Si et al., 2011; Lei et al., 2018)
—"tiempo hasta la cosecha" y "tiempo hasta la falla" son el mismo problema— y proviene del
**análisis de supervivencia clínico** (Ishwaran et al., 2008; Rizopoulos, 2011). Cualquier
proceso por lotes con una trayectoria que termina en un evento —fermentaciones, manufactura,
tiempos de entrega en logística— puede beneficiarse del mismo tratamiento. El trabajo aporta así
un caso replicable de cómo llevar la predicción dinámica de eventos, madura en otras disciplinas,
a un dominio productivo nuevo, con el cuidado metodológico que exige una muestra acotada.
