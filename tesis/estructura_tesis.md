# Estructura y storytelling de la tesis

> Documento maestro de trabajo. Plan de TF aprobado (oct-2026). Define el hilo conductor,
> la estructura de capítulos, el mapeo a lo ya producido y el plan de trabajo.
> Título (provisorio): *Modelos predictivos para la planificación de la producción en un
> cultivo hidropónico.*

---

## 1. El hilo conductor (storytelling)

**La tesis NO es "probé varios modelos sobre datos de albahaca". Es una historia con tesis
propia.** El relato en una frase:

> *Una operación hidropónica real genera un volumen enorme de datos pero planifica a ojo.
> Este trabajo demuestra que se puede predecir —con rigor y con la incertidumbre cuantificada—
> el tiempo de cosecha, adaptando herramientas maduras de otras disciplinas (análisis de
> tiempo al evento, predicción conforme) a un dominio nuevo. Y muestra, con honestidad
> metodológica, que el límite del desempeño no es el algoritmo sino la **escala real de la
> muestra** — un aporte transferible a cualquier problema con datos valiosos pero pocos casos
> independientes.*

### Los tres pilares del relato (todo capítulo debe empujar uno)
1. **Utilidad real.** Datos de una operación en funcionamiento; las predicciones se traducen
   en decisiones (cuándo cosechar, cuánto comprometer). No es un dataset académico.
2. **Rigor metodológico (el diferencial).** La unidad de generalización es la *cosecha*, no la
   fila; se valida agrupando por cosecha **y** temporalmente; se reportan intervalos con
   cobertura garantizada; se publican los **resultados negativos**. Esto es lo que distingue
   el trabajo de los que informan números inflados.
3. **El tamaño de muestra como protagonista.** No el modelo. La curva de aprendizaje, el
   fracaso del deep learning, la redundancia de features y la no-justificación de modelos
   "aperturados" son todas caras del mismo argumento.

### El giro narrativo (clave, y honesto)
Las "aristas" (proyección en T=0, rendimiento) **no son objetivos débiles**: son **evidencia a
favor de la tesis central**. Cada vez que un enfoque más complejo NO le gana a uno simple
(modelo estático aparte, fórmula sobre duración, forecasting, features derivadas, MLP), se
refuerza el mensaje: *con esta escala de datos, la prudencia le gana a la sofisticación.*
Los resultados negativos **se cuentan como hallazgos**, no se esconden.

---

## 2. Estructura de capítulos

> Estándar para tesis de maestría en explotación de datos. Entre corchetes, el insumo ya
> producido (TP o archivo en `Desarrollo/`).

**Resumen / Abstract + palabras clave** — [TP5]

**Cap. 1 — Introducción**
- 1.1 Contexto y motivación (hidroponía, agricultura de ambiente controlado, el dato poco explotado) — [TP1]
- 1.2 Problema de investigación — [TP1]
- 1.3 Objetivos (general + específicos, reencuadrados: eje = tiempo de cosecha; T=0 y rendimiento = exploratorios) — [TP2.2]
- 1.4 Aportes del trabajo — [TP3 §innovación]
- 1.5 Organización del documento

**Cap. 2 — Marco teórico y estado del arte**
- 2.1 ML para predicción en agricultura y rendimiento — [van Klompenburg, Liakos, Kamilaris]
- 2.2 Agricultura de ambiente controlado e hidroponía — [Kozai, Sambo, Benke & Tomkins, Saha]
- 2.3 Análisis de tiempo al evento y predicción dinámica — [Cox, Kaplan-Meier, Ishwaran, van Houwelingen, Rizopoulos]
- 2.4 Cuantificación de incertidumbre (predicción conforme) — [Vovk, Shafer, Lei, Romano, Angelopoulos]
- 2.5 Modelos de ML tabular (árboles, boosting) — [Breiman, Friedman, XGBoost, LightGBM, CatBoost]
- 2.6 Validación y escala de muestra — [Roberts, Cawley & Talbot, Kohavi, Bergstra & Bengio]
- 2.7 Trabajos relacionados y el vacío que llena esta tesis — [TP4.2]

**Cap. 3 — Materiales y métodos**
- 3.1 Caso de estudio (operación, albahaca, Bariloche, anonimizada) — [Metodología TP5]
- 3.2 Los datos: planilla manual + extractor por colores; estructura tanda/cosecha; enriquecimiento climático; planilla de rendimiento — [dataset/README, dataset_enrichment/, HGP xlsx]
- 3.3 Análisis exploratorio (EDA) — [`eda.py`, `eda_*.png`]
- 3.4 Diseño metodológico: landmarking; unidad de generalización = cosecha; validación agrupada vs temporal; métricas MAE/MedAE/RMSE — [`resumen.md`, `investigacion.md`]
- 3.5 Modelos, pipeline y optimización de hiperparámetros; predicción conforme; interpretabilidad (SHAP) — [`benchmark.py`, `campeon_catboost.py`]
- 3.6 Consideraciones éticas — [Informe de valor ético, TP5]

**Cap. 4 — Resultados**
- 4.1 Eje: predicción del tiempo de cosecha
  - 4.1.1 Progresión y viabilidad (dummy 8.2 → 2.8) — [`modelo_definitivo.py`]
  - 4.1.2 Challenge de tecnologías → CatBoost — [`challenge_modelos.md` §3, `benchmark.py`]
  - 4.1.3 Validación temporal vs agrupada — [`validacion_temporal.py`]
  - 4.1.4 Intervalos conformal — [`campeon_catboost.py`, §8]
  - 4.1.5 Factores accionables (SHAP) — [`shap_catboost.*`] ⚠️ falta PDP + recomendaciones (obj 4)
- 4.2 Proyección en T=0 (¿modelo estático aparte?) → no se justifica — [`challenge_modelos.md` §9]
- 4.3 Rendimiento (exploratorio) → estimación estacional ~22% — [`rendimiento_analisis.md`]
- 4.4 Resultados negativos como evidencia (GDD, features derivadas, forecasting, MLP)
- 4.5 ⭐ Curva de aprendizaje: el argumento de escala de muestra — ⚠️ PENDIENTE (`exp_curva_aprendizaje.png` existe, revisar/rehacer)

**Cap. 5 — Discusión**
- 5.1 Interpretación de los resultados; el tamaño de muestra como límite
- 5.2 Validez y limitaciones (shift train/test, target de rendimiento tentativo, una sola operación)
- 5.3 Transferencia a otros dominios (RUL, supervivencia clínica, procesos por lotes) — [TP4.1]

**Cap. 6 — Conclusiones y trabajo futuro** — [TP4.1 §ampliaciones]

**Referencias** (≥42, ya reunidas y verificadas con DOI)

**Anexos** — código, figuras, tablas de control de citas, diccionario de variables.

---

## 3. Mapeo: qué está hecho vs. qué falta

| Capítulo / sección | Estado | Insumo |
|---|---|---|
| Resumen, Objetivos, Problema, Solución, Aplicaciones, Transferencia, Trabajos relacionados | 🟢 escrito | TP1–TP5 (en el plan) |
| Marco teórico (ampliado) | 🟡 refs listas, falta redactar | 42 refs con DOI |
| Metodología | 🟢 redactada | TP5 Metodología |
| EDA | 🟢 hecho | `eda.py` + 5 figuras |
| Resultados eje (challenge, CatBoost, conformal, SHAP) | 🟢 hecho + documentado | `challenge_modelos.md` |
| Resultado T=0 (obj 2) | 🟢 hecho + documentado | `challenge_modelos.md` §9 |
| Resultado rendimiento (obj 3) | 🟢 hecho + documentado | `rendimiento_analisis.md` |
| **Curva de aprendizaje** (keystone del argumento) | 🟢 HECHO | `curva_aprendizaje.py/.png/.csv` (4.80→3.13, sigue bajando) |
| **Obj 4: PDP + recomendaciones accionables** | 🟢 HECHO | `accionables_pdp.*`, `accionables_analisis.md` (efecto <0.1 d) |
| Predicción dinámica formal de supervivencia (A2) | 🟡 opcional | RSF hecho |
| Redacción integradora (capítulos 1–6) | 🟢 BORRADOR v1 | `tesis/01..06_*.md` + `Tesis - borrador v1.docx` |

**ESTADO (02-10-2026): borrador completo v1 de la tesis ensamblado** (`tesis/Tesis - borrador v1.docx`, ~6.500 palabras, 6 capítulos + resumen + 42 referencias). Capítulos en `tesis/01..07_*.md`. Faltan: pulir redacción, insertar figuras en el .docx, y (opcional) predicción dinámica formal de supervivencia.

---

## 4. Plan de trabajo propuesto (orden sugerido)

1. **Cerrar los 2 análisis faltantes clave** (son evidencia central):
   - ⭐ **Curva de aprendizaje** (MAE vs nº de cosechas) → prueba el argumento de escala.
   - **Obj 4**: dependencia parcial (PDP) + traducir SHAP a recomendaciones con discusión causal.
2. **Consolidar el capítulo de Resultados** (ya casi todo producido) en narrativa.
3. **Redactar Metodología** ampliada (desde TP5 + EDA + diseño).
4. **Redactar Marco teórico** (desde las 42 refs).
5. **Redactar Introducción, Discusión y Conclusiones** (desde TP1–TP4 + hallazgos nuevos).
6. **Ensamblar** el documento final y exportar a `.docx` para entrega.

**Criterio de escritura:** cada sección empuja al menos uno de los tres pilares; los resultados
negativos se cuentan como hallazgos; nada de números sin su validación temporal al lado.
