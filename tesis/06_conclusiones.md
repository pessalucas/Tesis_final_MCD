# Capítulo 6 — Conclusiones y trabajo futuro

## 6.1 Conclusiones

Este trabajo desarrolló, sobre datos reales de una operación hidropónica de albahaca, un sistema
predictivo para la planificación de la producción, con la estimación del tiempo de cosecha como
eje. Las principales conclusiones son:

1. **Es viable predecir el tiempo de cosecha con utilidad operativa.** El modelo campeón
   (CatBoost) alcanza un MAE de ≈2.9 días en validación temporal (frente a 8.4 del predictor
   trivial), re-estimable día a día, y con intervalos de cobertura calibrada de ±5 días al 80 %.
   Esto habilita comprometer fechas con un riesgo explícito y reducir las cosechas parciales de
   emergencia.

2. **El encuadre de tiempo al evento con re-estimación dinámica es adecuado y transferible.** Un
   único modelo dinámico cubre tanto la planificación estratégica (estimación al inicio) como la
   táctica (ajuste sobre la marcha); no se justifica un modelo estático aparte.

3. **El límite del desempeño es la escala de la muestra, no el modelo.** La curva de aprendizaje
   muestra que el error seguiría bajando con más cosechas; los métodos de árboles quedan parejos
   entre sí; el aprendizaje profundo falla por falta de datos; y las features derivadas, el
   modelo estático y el *forecasting* no aportan. La contribución metodológica central es tratar
   esta limitación con rigor —unidad de generalización = cosecha, validación agrupada y temporal,
   intervalos conformal— y reportarla con honestidad.

4. **Los factores controlables por el operador tienen un efecto marginal despreciable sobre la
   duración**, gobernada por la estación y la luz; las asociaciones halladas no deben leerse como
   causalidad.

5. **El rendimiento es estimable de forma estacional (~22 % de error), pero los datos actuales no
   justifican un modelo dedicado** (objetivo exploratorio): se requiere más historia y un registro
   por cosecha en origen.

En conjunto, el trabajo aporta no solo predicciones útiles para una operación real, sino un
**ejemplo replicable de cómo obtenerlas con rigor cuando los datos son valiosos pero limitados**.

## 6.2 Trabajo futuro

- **Acumular más cosechas** (la palanca de mayor retorno según la curva de aprendizaje) y, para
  el rendimiento, **registrarlo por cosecha en origen**.
- **Incorporar variables de la planta** (morfología: número de hojas, cobertura; eventualmente
  imágenes), de alto poder predictivo según la literatura, en especial para la estimación
  temprana.
- **Predicción dinámica formal de supervivencia** (landmarking de supervivencia) para producir
  curvas de días-restantes día a día, complementando la regresión puntual.
- **Pronóstico estacional backtesteable** como feature de "expectativa pública", evaluando su
  aporte real antes de adoptarlo.
- **Transferencia de aprendizaje** desde otras instalaciones y cultivos para mejorar la
  generalización más allá de una única operación.
- **Validación experimental** de las recomendaciones de manejo, para pasar de asociación a
  causalidad.
