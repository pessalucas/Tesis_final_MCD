# pronostico_smn · Agente extractor del Pronóstico Climático Trimestral (SMN)

Convierte el **pronóstico estacional NO estructurado** (texto/reportes del SMN, o notas de
diarios que lo publican) en una **señal mensual estructurada** de terciles para Bariloche,
lista para cruzar por FECHA con el resto del enrichment. Resuelve el problema que ninguna
API gratis daba: un **outlook estacional backtesteable** (si se ingestan los reportes pasados).

## La fuente

**Pronóstico Climático Trimestral (PCT) del SMN** — se emite **cada mes**, da **terciles**
(sobre / normal / bajo lo normal) de **temperatura y precipitación** por región, a 90 días.
Para Bariloche interesan las menciones a "Patagonia (norte)", "cordillera", "Río Negro",
"Neuquén". Se basa en modelos numéricos internacionales (NMME/IRI).

## Dos caminos de ingesta (ambos backtesteables)

1. **Estructurado y robusto — IRI/NMME**: el IRI Data Library publica los terciles
   mensuales de temp/precip desde **2001**, grilla 1°, descargables por punto (OPeNDAP).
   Es la vía sólida si se quiere el dato numérico directo, sin parseo de texto.
2. **No estructurado (este agente) — texto del PCT / notas**: parsear el reporte del SMN
   o las notas (Río Negro, Infobae, La Nación) que lo publican mes a mes, y extraer la
   señal de terciles. Es la vía "convertir data no estructurada en dato".

## Esquema de salida

Una fila por mes de emisión, cruzable por FECHA (vale para el trimestre pronosticado):

| Campo | Valores | Qué es |
|---|---|---|
| `temp_tercil` | +1 / 0 / −1 | temperatura sobre / normal / bajo lo normal |
| `prec_tercil` | +1 / 0 / −1 | precipitación sobre / normal / bajo lo normal |
| `region_match` | texto | región de la que se tomó la señal |
| `confianza` | alta / media | si matcheó región específica de Bariloche |

## Componentes

- `parser.py` — **core del agente**: `parse_pct(texto) -> dict`. Prototipo por reglas
  (regex sobre frases típicas en español). Probado con texto real (JJA-2026, verano
  25/26). **Limitación conocida**: al ser por reglas, a veces mezcla la señal de temp con
  la de precip cuando están en la misma oración (ej. "temperaturas superiores... y
  precipitaciones dentro de lo normal"). En producción/tesis, reemplazar el motor de
  reglas por un **LLM** que parsee cualquier redacción de forma robusta — la interfaz
  (texto → dict) queda igual.

## Pendiente para completar la ingesta (past + future)

1. **Fuente de reportes pasados**: encontrar el archivo del PCT (el sitio del SMN bloquea
   fetch directo → 402). Alternativas: archivo de notas de diarios (buscador por fecha),
   o directamente IRI/NMME estructurado (camino 1) para saltear el parseo.
2. **Loop de ingesta**: por cada mes 2022→hoy, bajar el reporte/nota, `parse_pct`, y
   guardar `pronostico_smn_mensual.csv` (FECHA_emision, trimestre, temp_tercil, prec_tercil).
3. **Futuro (producción)**: mismo parser sobre el PCT del mes corriente.
4. **Cruce y evaluación**: mapear cada mes de emisión a las cosechas cuyo ciclo cae en el
   trimestre pronosticado; medir si `temp_tercil`/`prec_tercil` aportan sobre el clima
   diario. Expectativa realista: aporte incierto (baja habilidad estacional en la Patagonia),
   pero ahora **medible y backtesteable**.
