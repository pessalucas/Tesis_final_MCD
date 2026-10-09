"""Ingesta del PCT del SMN -> CSV estructurado mensual (agente extractor, end-to-end).

Toma una lista de reportes (fecha de emisión + trimestre pronosticado + texto crudo),
los pasa por el parser (texto no-estructurado -> terciles) y produce
`pronostico_smn_mensual.csv`, cruzable por fecha con el resto del enrichment.

El texto puede venir de: (a) un archivo/lista provista, (b) scraping de notas/reportes.
Acá se demuestra con reportes de ejemplo; la fuente de reportes PASADOS (2022→) es el
pendiente (SMN bloquea fetch directo; alternativa: IRI/NMME estructurado o archivo de notas).
"""
import os
import pandas as pd
from parser import parse_pct

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pronostico_smn_mensual.csv")

def ingest(reportes: list[dict]) -> pd.DataFrame:
    """reportes: [{'fecha_emision','trimestre','texto'}...] -> DataFrame estructurado."""
    filas = []
    for r in reportes:
        s = parse_pct(r["texto"])
        filas.append({"fecha_emision": r["fecha_emision"], "trimestre": r["trimestre"],
                      **{k: s[k] for k in ["temp_tercil","prec_tercil","region_match","confianza"]}})
    df = pd.DataFrame(filas)
    df["fecha_emision"] = pd.to_datetime(df["fecha_emision"])
    return df.sort_values("fecha_emision")

if __name__ == "__main__":
    # DEMO con reportes de ejemplo (en producción: lista completa 2022->hoy)
    reportes = [
        {"fecha_emision":"2026-05-20","trimestre":"JJA-2026",
         "texto":"En el centro-norte de la Patagonia se esperan precipitaciones por encima "
                 "de lo normal, mientras que las temperaturas se mantendrían superiores a lo "
                 "normal en gran parte de la Patagonia."},
        {"fecha_emision":"2025-11-20","trimestre":"DEF-2025/26",
         "texto":"Para el norte de la Patagonia, temperaturas superiores a lo normal y "
                 "precipitaciones dentro de lo normal."},
        {"fecha_emision":"2025-05-20","trimestre":"JJA-2025",
         "texto":"Se prevén temperaturas inferiores a lo normal y lluvias por debajo de lo "
                 "normal en la cordillera de Río Negro y Neuquén."},
    ]
    df = ingest(reportes)
    df.to_csv(OUT, index=False)
    print("Ingesta -> CSV estructurado:")
    print(df.to_string(index=False))
    print(f"\nGuardado: {OUT}")
