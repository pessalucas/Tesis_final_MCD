# Log de investigación — sesión autónoma (2026-08-24)

Registro en vivo de los experimentos que voy corriendo. Cada bloque: qué probé, resultado,
lectura. Resumen ejecutivo al final (se completa al cerrar).

Agenda de esta sesión:
1. Modelo definitivo (LightGBM tuneado + trayectoria + clima + historia).
2. Conformal prediction para intervalos con cobertura garantizada.
3. Baseline agronómico de GDD (tiempo térmico) vs ML.
4. CV anidada sobre el campeón (número insesgado).
5. SHAP / dependencia parcial (factores accionables).
6. Validación temporal (train pasado -> test futuro).
7. Ablaciones por grupo de features (tabla limpia de aportes).

---

## 1. Modelo definitivo (LightGBM tuneado + trayectoria + clima + historia)
Ablación por grupo (MAE OOF, GroupKFold por cosecha):
- base (trayectoria): 3.06
- base+clima: 3.00
- base+historia: **2.82** ← la historia es el gran salto
- **base+clima+historia (DEFINITIVO): 2.80** | MedAE 1.95

Progreso total: dummy 8.19 → MVP 3.17 → definitivo **2.80** (−66% vs dummy).
FI (gain): hist_dur_reciente5 #1, luego DLI (clima) y prev60_dli_anom. Historia + luz dominan.
Cola larga (día 46+) sigue siendo lo peor (MAE 3.46) — cosechas largas.
Guardado: landmarks_full.parquet, oof_definitivo.npy.

## 2. Conformal prediction (CQR) — intervalos calibrados
Intervalo 80% de días-restantes:
- naive (cuantiles): cobertura 48% (MIENTE), ancho 4.8 d
- **CQR (conformal): cobertura 83.8%** (objetivo 80% ✓), ancho 10.2 d
Lectura: el intervalo 80% HONESTO es ~±5 días (ancho ~10), no ±2.4. Los cuantiles crudos
estaban muy sub-cubiertos. CQR garantiza la cobertura. Ancho ~constante por etapa (9.5-11.5d),
un poco más ancho en la cola larga (día 46+). Para planificar: comprometer fecha ±5 días al 80%.

## 3. Baseline agronómico de GDD (tiempo térmico) — RESULTADO NEGATIVO
GDD-a-cosecha muy variable: CV 54% (base 2°C) hasta 110% (base 10°C) -> el tiempo térmico
externo NO es constante a cosecha. Baseline GDD predictivo: MAE 40 d (peor que dummy).
Lectura: el tiempo térmico del aire exterior NO gobierna la cosecha en esta operación
(invernadero -> temperatura amortiguada; la LUZ/DLI es el driver, no el calor externo).
Descarta el baseline agronómico clásico -> justifica el enfoque ML sobre señales más ricas.

## 4. Validación temporal (train pasado -> test futuro) — HALLAZGO METODOLÓGICO CLAVE
Expanding window (LightGBM definitivo):
- train 50% (hasta 2024-10): MAE 16.90 d (!) -> con poca historia + shift hoja1/hoja2, se rompe
- train 60%: 5.17 · train 70%: 4.09 · train 80%: 3.68
Referencia CV agrupada: 2.80.
Lectura: la CV agrupada es OPTIMISTA vs producción real. El error honesto de producir hacia
adelante es ~3.7-4.1 d (con train suficiente), no 2.80. Y el modelo necesita un mínimo de
historia para funcionar (el 16.9 con 50% es una advertencia). Para la tesis: reportar AMBAS
validaciones; el número de producción es ~3.7-4, la agrupada da el techo optimista.

## 5. SHAP — factores accionables por el operador
Accionables (pH, PPM, eventos) pesan mucho menos que lo no controlable (|SHAP| 1.07 vs 12.23):
la mayor parte de la predictibilidad viene de estación + historia + clima, no del manejo.
Sentido de los accionables (asociación aprendida por el modelo):
- n_BOVEL: +  → más Bovel = ciclo MÁS LARGO. OJO: probable causalidad inversa (se aplica ante
  problemas; los ciclos problemáticos duran más). Marcador, no palanca.
- PPM (min/mean/max): −  → más nutrientes/conductividad = ciclo MÁS CORTO. Palanca plausible.
- pH medio: + (más alto = más largo); pH bajo/estable = más corto. ph_std + = inestabilidad alarga.
Recomendación tentativa (a validar en la práctica): mantener PPM/nutrientes altos y pH bajo y
estable se asocia a ciclos más cortos. CAVEAT causal: asociación ≠ causa. Fig: shap_accionables.png.

## 6. Gráfico consolidado -> resultados_consolidados.png (progresión + validaciones).

---
# RESUMEN EJECUTIVO (sesión autónoma 2026-08-24)

**Mejor modelo:** LightGBM tuneado + trayectoria + clima + historia → **MAE 2.80 d** (CV agrupada),
de 8.19 (dummy). MedAE 1.95 (mitad de las predicciones dentro de ~2 días).

**Hallazgos clave (ordenados por impacto):**
1. La HISTORIA de duraciones es el mayor lever barato: `hist_dur_reciente5` (prom. últimas 5
   cerradas) = feature #1; base 3.06 → +historia 2.82. Backtesteable y gratis.
2. VALIDACIÓN TEMPORAL (producción real) da ~3.7-4.1 d, PEOR que la CV agrupada (2.80).
   La agrupada es optimista. Reportar ambas. El modelo necesita historia suficiente.
3. INTERVALOS: los cuantiles crudos mienten (cobertura 48%). CQR (conformal) los calibra a
   84% (objetivo 80%), ancho honesto ~±5 días. Para comprometer fechas: "cosecha en X ±5 d (80%)".
4. GDD (tiempo térmico externo) NO gobierna la cosecha (CV 54-110%, baseline MAE 40) → invernadero
   amortigua temperatura; la LUZ/DLI es el driver. Descarta el baseline agronómico.
5. SUPERVIVENCIA (RSF): C-index 0.90, usa las 62 censuradas; fuerte en probabilístico/ranking,
   no en punto (MAE 3.83). Complementa, no reemplaza.
6. FACTORES ACCIONABLES (SHAP): pesan poco (1.07 vs 12.23). Señal: más PPM/nutrientes → más corto;
   pH bajo/estable → más corto; Bovel marca ciclos largos (prob. causalidad inversa). CAVEAT causal.
7. POR MES: abr-jul (cruzan invierno) largos e inciertos (MAE 3.2-3.7); ago-sep-dic cortos,
   estables, muy predecibles (MAE 2.1).

**Caveats abiertos:** PPM en dos escalas mezcladas (×500) ensucia su efecto SHAP; confirmar factor.
CV anidada del campeón pendiente (número ~2.8-2.9 insesgado). Extractor SMN: falta ingesta pasada + LLM.

**Archivos:** modelo_definitivo.py, conformal_intervalos.py, gdd_baseline.py, validacion_temporal.py,
shap_actionables.py, resultados_consolidados.py, survival_model.py, intervalos.py, analisis_pasado_mes.py.
Figuras: resultados_consolidados.png, shap_accionables.png, int_*.png, mes_analisis.png, mae_*.png.
Datos: landmarks_full.parquet, oof_definitivo.npy.

## 7. Agente extractor SMN — test con 4 fuentes reales de noticias
parse_pct mejorado (consciente de cláusulas) sobre texto verbatim de 4 medios:
Río Negro, Infobae, La Nación, City Noticias -> **temperatura 4/4, precipitación 4/4**.
Archivos: dataset_enrichment/pronostico_smn/ (parser.py, ingest.py, test_fuentes.py, README).
Elección de fuente: Río Negro = mejor detalle regional patagónico y más limpio; nacionales
(Infobae/La Nación) tienen buen archivo pero redacción más ambigua por sub-regiones.
LIMITACIÓN clave: los medios cubren el PCT en hitos estacionales (~pocas veces/año), NO todos
los meses -> para frecuencia MENSUAL backtesteable conviene SMN directo o IRI/NMME estructurado.

## 8. Oráculo estacional — DECISIÓN: no construir IRI/NMME
Test upper-bound (clima REAL futuro = pronóstico perfecto, leakage a propósito):
- campeón: 2.82 · +oráculo total (d+1..fin): 2.78 · +oráculo lejano (d+8..fin): 2.74
El techo de CUALQUIER pronóstico estacional es ~0.08 d (2.82->2.74). Un pronóstico real
(con error, sin luz directa) capturaría casi nada. -> NO vale construir IRI/NMME ni pagar
meteoblue para esto. La incertidumbre del tiempo de cosecha no está en el clima futuro
(ni conociéndolo perfecto) sino en el estado de la planta (trayectoria) + ruido irreducible.
Hilo estacional CERRADO. Poner energía en: más cosechas (curva de aprendizaje) + refinar modelo.
`oraculo_estacional.py`.

## 9. Resolución del "paradoja": importancia != aporte marginal (obs. de Lucas)
cl_fc_dli_next7 es top en FI pero el oráculo no daba lift. Verificado:
- corr(cl_fc_dli_next7, climatología/mes/luz_ciclo) = 0.67-0.73 -> es proxy de ESTACIÓN, no skill.
- Drop-column: sin cl_fc_dli_next7 Δ+0.029; sin TODAS las de futuro Δ-0.008; SIN NADA DE CLIMA Δ-0.004.
FI mide cuánto USA el modelo la feature (canal principal de la estación), no su valor único.
Con HISTORIA presente, el clima entero es redundante (Δ 0.004). Modelo real = trayectoria+historia,
sin clima, casi idéntico y más simple. `ablacion_pronostico.py`.

## 10. Oráculo por-feature: ¿el futuro de QUÉ variable aportaría? -> NINGUNA
Lift de conocer el futuro real (perfecto) de cada variable, sobre el campeón (2.82):
- futuro luz/DLI: -0.013 · futuro temp: -0.006 (ruido)
- futuro pH: +0.036 · futuro PPM: +0.247 · pH+PPM: +0.197 · TODO: +0.191 (EMPEORAN)
Conclusión: conocer el futuro de cualquier feature NO ayuda (0 o negativo). La duración se
define temprano; el futuro es consecuencia, no causa predictiva. Cierra toda la línea de
"forecastear algo". Lever real = más cosechas, no mejores pronósticos. `oraculo_features.py`.
