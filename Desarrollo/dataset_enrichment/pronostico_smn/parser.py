"""Agente extractor · Pronóstico Climático Trimestral (SMN) — texto -> dato estructurado.

CORE del extractor: toma el texto NO ESTRUCTURADO de un pronóstico climático trimestral
(reporte del SMN o nota de diario que lo publica) y lo convierte en una señal
ESTRUCTURADA de terciles para la región de Bariloche/Patagonia norte:

    temp_tercil, prec_tercil  ∈ {+1 sobre lo normal, 0 normal, -1 bajo lo normal}

Esa señal, mensual, se cruza por FECHA con el resto del enrichment y sirve como feature
del "outlook estacional" (backtesteable si se ingestan los reportes pasados).

Este prototipo usa reglas (regex sobre frases típicas en español). En producción/tesis,
un LLM haría el parseo robusto de cualquier redacción; la interfaz (texto -> dict) es la
misma. La ingesta de TODOS los reportes pasados/futuros se describe en el README.

Nota: el SMN reporta por regiones. Para Bariloche interesan las menciones a
"Patagonia", "Patagonia norte", "cordillera", "Río Negro", "Neuquén".
"""
import re
import unicodedata

REGIONES_BRC = ["patagonia norte", "norte de la patagonia", "cordillera", "rio negro",
                "neuquen", "patagonia"]  # de más específica a más general

def _norm(t: str) -> str:
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode().lower()
    return re.sub(r"\s+", " ", t)

# frases -> tercil
SOBRE = ["superior", "por encima", "mas calid", "mas caluros", "mas lluvi", "superiores",
         "mayores a lo normal", "por arriba"]
BAJO  = ["inferior", "por debajo", "mas fri", "menos lluvi", "menores a lo normal", "deficit"]
NORMAL = ["normal", "dentro de lo normal", "sin desvios", "categoria normal"]

def _tercil(frag: str) -> int:
    f = _norm(frag)
    if any(s in f for s in SOBRE): return 1
    if any(s in f for s in BAJO): return -1
    if any(s in f for s in NORMAL): return 0
    return 0  # default: sin señal -> normal

def parse_pct(texto: str) -> dict:
    """Extrae {temp_tercil, prec_tercil, region_match, confianza} del texto del PCT.

    Parseo CONSCIENTE DE CLÁUSULAS: parte el texto en cláusulas (por ; , . y 'mientras',
    ' y ') y asigna a cada variable el tercil de LA cláusula que la menciona, para no
    contaminar la señal de precipitación con la de temperatura vecina (y viceversa).
    """
    t = _norm(texto)
    region = next((r for r in REGIONES_BRC if r in t), None)
    clausulas = re.split(r"[;,\.]| mientras | y (?=(?:las|la|el|los|se|precipit|temperat|lluvi))", t)

    def señal(claves):
        cands = [c for c in clausulas if any(k in c for k in claves)]
        # priorizar cláusulas que además nombran la región de Bariloche/Patagonia
        reg_cands = [c for c in cands if region and (region in c or "patagonia" in c)]
        for c in (reg_cands or cands):
            terc = _tercil(c)
            if terc != 0 or "normal" in c:
                return terc
        return 0

    temp = señal(["temperatura", "termic"])
    prec = señal(["precipitacion", "lluvia"])
    return {"temp_tercil": temp, "prec_tercil": prec,
            "region_match": region or "(sin región explícita)",
            "confianza": "alta" if region and region != "patagonia" else "media"}


if __name__ == "__main__":
    # pruebas con texto real (parafraseado de notas del PCT del SMN, ago-2026 y verano 25/26)
    ejemplos = [
        ("JJA-2026",
         "El SMN emitió el pronóstico climático trimestral para junio-julio-agosto. "
         "En el centro-norte de la Patagonia se esperan precipitaciones por encima de lo "
         "normal, mientras que las temperaturas se mantendrían superiores a lo normal en "
         "gran parte de la Patagonia."),
        ("DEF-2025/26 (verano)",
         "El pronóstico trimestral anticipa un verano más cálido y húmedo en gran parte "
         "del país. Para el norte de la Patagonia, temperaturas superiores a lo normal y "
         "precipitaciones dentro de lo normal."),
        ("caso frío seco",
         "Se prevén temperaturas inferiores a lo normal y lluvias por debajo de lo normal "
         "en la cordillera de Río Negro y Neuquén durante el trimestre."),
    ]
    for etq, txt in ejemplos:
        print(f"{etq:22s} -> {parse_pct(txt)}")
