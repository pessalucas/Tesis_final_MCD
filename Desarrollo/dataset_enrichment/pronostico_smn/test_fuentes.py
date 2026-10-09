"""Test del agente extractor con TEXTO REAL de distintas fuentes de noticias.

Cada fuente publica el mismo pronóstico trimestral del SMN, pero redactado distinto.
Objetivo: ver de cuál fuente el parser extrae la señal más limpia para Bariloche
(Patagonia norte) -> ayuda a elegir qué fuente ingestar de forma sistemática.

Textos: frases verbatim extraídas de los artículos (WebFetch), tramo Patagonia.
"""
from parser import parse_pct

# (fuente, trimestre, esperado_temp, esperado_prec, texto_real)
FUENTES = [
    ("Río Negro", "trimestral", +1, +1,
     "En una zona de la Patagonia habrá temperaturas superiores a las habituales. "
     "Las precipitaciones serán superior a lo normal en el centronorte de Patagonia; "
     "en el sur del Litoral y sur de Patagonia el rango se mantendrá dentro de lo normal."),
    ("Infobae", "ASO-2026", 0, 0,
     "En Cuyo, La Pampa, Buenos Aires y el centro y norte de la Patagonia las temperaturas "
     "se mantendrían dentro de los valores habituales del trimestre. En la Patagonia, el este "
     "muestra precipitaciones excediendo los valores históricos; el sudoeste de la Patagonia "
     "registraría precipitaciones dentro de los valores normales para el trimestre."),
    ("La Nación", "DEF-2025/26", +1, 0,
     "En el centro y norte de la Patagonia el organismo estima entre 50% y 55% de probabilidad "
     "de que la temperatura media se ubique por encima del promedio histórico. Una probabilidad "
     "cercana al 40% de que las lluvias se mantengan dentro de los valores normales en el oeste "
     "y sur de la Patagonia."),
    ("City Noticias", "JJA-2024", -1, -1,
     "Gran parte de Patagonia tendría temperaturas inferiores a las normales. Será un invierno "
     "con pocas o escasas lluvias, especialmente en Patagonia, con lluvias inferiores a las "
     "normales para la estación del año."),
]

TERC = {1: "sobre(+1)", 0: "normal(0)", -1: "bajo(-1)"}
print(f"{'FUENTE':14s} {'TRIM':12s} {'temp (pred/esp)':>18s} {'prec (pred/esp)':>18s} {'región':>22s}")
print("-"*90)
ok_t = ok_p = 0
for fuente, trim, et, ep, txt in FUENTES:
    s = parse_pct(txt)
    mt = "OK" if s["temp_tercil"] == et else "XX"; mp = "OK" if s["prec_tercil"] == ep else "XX"
    ok_t += s["temp_tercil"] == et; ok_p += s["prec_tercil"] == ep
    print(f"{fuente:14s} {trim:12s} "
          f"{TERC[s['temp_tercil']]:>10s}/{TERC[et]:<7s}{mt} "
          f"{TERC[s['prec_tercil']]:>10s}/{TERC[ep]:<7s}{mp} {s['region_match']:>22s}")
print("-"*90)
print(f"Aciertos: temperatura {ok_t}/{len(FUENTES)} | precipitación {ok_p}/{len(FUENTES)}")
