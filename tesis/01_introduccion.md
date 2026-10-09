# Capítulo 1 — Introducción

## 1.1 Contexto y motivación

La agricultura de ambiente controlado, y en particular la **hidroponía**, es una modalidad de
producción en expansión: cultiva plantas sin suelo, con las raíces en una solución nutritiva
cuya composición, temperatura y demás parámetros se regulan de manera activa. Permite producir
alimentos de forma intensiva, con menor uso de agua y de espacio, y cerca de los centros de
consumo (Benke & Tomkins, 2017; Sambo et al., 2019). Estos sistemas están fuertemente
instrumentados y generan un volumen considerable de datos: cada cosecha es un proceso que dura
varias semanas y se registra día a día, con múltiples variables ambientales y de manejo.

Sin embargo, buena parte de ese dato permanece **poco explotado**. La planificación de la
producción se apoya sobre todo en la experiencia del operador, con reglas aprendidas en la
práctica que funcionan de manera despareja y que son difíciles de transmitir o de sostener
cuando la operación crece (Wolfert et al., 2017; Kamilaris et al., 2017). El aprendizaje
automático se ha consolidado como una herramienta de apoyo a la decisión en agricultura, capaz
de anticipar resultados productivos y de orientar la gestión del ciclo (Liakos et al., 2018;
van Klompenburg et al., 2020), lo que abre la oportunidad de convertir esos datos en decisiones
mejor fundadas.

Este trabajo se sitúa en ese cruce entre la ciencia de datos y la producción hidropónica, sobre
un caso **real y en funcionamiento**: una operación comercial de albahaca que lleva registradas
cerca de doscientas cosechas históricas con seguimiento diario. Que los datos provengan de una
operación genuina —y no de un conjunto académico o sintético— es lo que ancla el trabajo en un
problema concreto, con un usuario que espera resultados aplicables.

## 1.2 Problema de investigación

El problema general es la **dificultad para anticipar, con datos y no solo con experiencia, los
resultados de cada ciclo de producción**. Hoy existe incertidumbre sobre tres cuestiones clave
para operar: cuándo estará lista una cosecha, cuánto rendirá, y qué condiciones del proceso
influyen realmente en esos resultados.

La dificultad de fondo es que el resultado de una cosecha no se define en un instante: es el
producto de una trayectoria de varias semanas en la que muchas variables cambian día a día. El
tiempo hasta la cosecha, por ejemplo, no es un número fijo conocido de antemano, sino algo que
se va perfilando a medida que el proceso avanza y que idealmente debería poder actualizarse con
la información que se acumula.

Las consecuencias son operativas y económicas. Si no se anticipa bien el tiempo de cosecha, se
desajusta la planificación: se asumen compromisos comerciales que luego cuesta cumplir, o se
subutiliza la capacidad instalada. Un caso especialmente costoso, observado en la operación, es
la fuerte variabilidad de la cosecha a lo largo del año: para cumplir con la demanda
comprometida, muchas veces hay que salir con **cosechas parciales** —levantar la producción
antes de su punto óptimo—, lo que desaprovecha gran parte del potencial productivo. La falta de
anticipación se traduce así, directamente, en pérdida de valor.

## 1.3 Objetivos

**Objetivo general.** Desarrollar modelos predictivos a partir de los datos de una operación
hidropónica real, con el fin de mejorar la planificación y la eficiencia de su producción.

**Objetivos específicos.**
1. *(Eje)* **Estimar de forma dinámica el tiempo restante hasta la cosecha**, actualizable día a
   día durante el ciclo, con el fin de anticipar la disponibilidad de la producción y reducir las
   cosechas parciales que desaprovechan el potencial productivo.
2. **Cuantificar la incertidumbre** de esas estimaciones mediante intervalos con cobertura
   calibrada, para poder comprometer fechas y volúmenes con un riesgo explícito.
3. **Identificar las variables controlables por el operador** con mayor influencia sobre los
   resultados, y derivar recomendaciones prácticas, discutiendo sus límites causales.
4. *(Exploratorio)* **Explorar la viabilidad** de proyectar tempranamente, al inicio del ciclo,
   el tiempo de cosecha y el rendimiento (cantidad de plantas), para evaluar en qué medida la
   planificación puede sustentarse en datos desde el comienzo.

## 1.4 Aportes del trabajo

El grado de innovación no radica en inventar un algoritmo nuevo, sino en la **aplicación y
combinación poco frecuentes de métodos consolidados sobre un problema real**, con tres aportes
centrales:

1. **Encuadre del tiempo de cosecha como un problema de tiempo al evento con re-estimación
   dinámica** (mediante *landmarking*), un enfoque maduro en el análisis de supervivencia y en el
   mantenimiento predictivo, pero —hasta donde se relevó— novedoso en la estimación del tiempo de
   cosecha de un cultivo hidropónico real.
2. **Rigor metodológico acorde a la escala real de la muestra.** El trabajo distingue el número
   de filas (miles de registros diarios) del número de unidades independientes (las cosechas),
   valida agrupando por cosecha y, además, de manera **temporal** (entrenar con el pasado, predecir
   el futuro), y reporta intervalos con garantía de cobertura en lugar de métricas puntuales.
3. **Honestidad metodológica como resultado.** Se reportan explícitamente los **resultados
   negativos** (el tiempo térmico no gobierna el ciclo; las features derivadas no aportan; un
   modelo estático dedicado no se justifica; el forecasting es inviable con la historia
   disponible), porque cada uno fortalece la conclusión de fondo: **con esta escala de datos, el
   límite del desempeño es la muestra, no el modelo.**

## 1.5 Organización del documento

El Capítulo 2 presenta el marco teórico y el estado del arte. El Capítulo 3 describe el caso de
estudio, los datos, el análisis exploratorio y el diseño metodológico. El Capítulo 4 reporta los
resultados, organizados en torno al eje (tiempo de cosecha) y a las aristas exploratorias. El
Capítulo 5 discute los hallazgos, su validez y su transferencia a otros dominios. El Capítulo 6
concluye y propone líneas de trabajo futuro.
