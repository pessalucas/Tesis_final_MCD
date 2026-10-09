# Objetivo 4 — Factores accionables: SHAP + dependencia parcial + recomendaciones

> Cierre del objetivo 4. Scripts: `shap_catboost.py`/`campeon_catboost.py` (SHAP),
> `accionables_pdp.py` (PDP). Figuras: `shap_catboost.png`, `accionables_pdp.png`.

## 1. Pregunta
¿Qué variables **controlables por el operador** (pH, PPM, eventos de manejo) influyen en la
duración del ciclo, y cuánto se puede acortar interviniendo sobre ellas?

## 2. Importancia global (SHAP, campeón CatBoost)
La predicción está dominada por variables **no controlables**: historia de duraciones
(`hist_dur_reciente5`, `hist_dur_mismomes`), estación (`mes_*`) y luz (`cl_*_dli`). El único
evento de manejo en el top-20 es `n_BOVEL`. En magnitud, los accionables pesan mucho menos que
el bloque estación+historia+clima.

## 3. Efecto marginal (dependencia parcial, PDP)
Moviendo cada variable accionable de su percentil 5 al 95 (resto fijo), el cambio en la duración
**predicha** es:

| Variable accionable | Δ duración (p5→p95) |
|---|---:|
| n_BOVEL | +0.8 d |
| ppm_mean | −0.1 d |
| ppm_min | −0.1 d |
| n_AGREGADO_SALES | ~0 d |
| dias_desde_AGREGADO_SALES | ~0 d |
| ph_std | ~0 d |
| ph_mean | ~0 d |

## 4. Lectura y recomendaciones
- **El manejo fino tiene un efecto marginal despreciable (<0.1 d) sobre la duración del ciclo.**
  La duración está gobernada por factores no controlables (estación/luz) e historia.
- `n_BOVEL` muestra el mayor efecto (+0.8 d) pero con **causalidad inversa**: el Bovel se aplica
  ante ciclos problemáticos, no los causa; es un **marcador**, no una palanca.
- **Recomendación honesta:** no hay evidencia de que ajustar pH/PPM/sales dentro de los rangos
  observados acorte el ciclo de forma relevante. El valor del sistema está en **anticipar**
  (predecir y planificar), no en **intervenir** sobre estas variables para acelerar la cosecha.
- Las asociaciones (más PPM → ligeramente más corto; pH bajo/estable → ligeramente más corto)
  son débiles y deben validarse experimentalmente antes de traducirse en prácticas.

## 5. Conclusión del objetivo 4
Objetivo cumplido: se identificaron las variables accionables y se **cuantificó su influencia
(baja)** con SHAP y PDP. El aporte es tanto el método (importancia por grupos + PDP + caveat
causal) como el hallazgo honesto de que, en esta operación, el poder predictivo del manejo sobre
la duración es limitado.
