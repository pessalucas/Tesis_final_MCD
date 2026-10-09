"""Gráfico consolidado de resultados — para la tesis.
Progresión de modelos (MAE) + comparación de esquemas de validación.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# progresión de modelos (MAE OOF, GroupKFold por cosecha)
modelos = ["Dummy\n(promedio)", "Prom. por mes\n(causal)", "Solo\nestáticas",
           "MVP\n(trayectoria)", "+ clima", "+ historia", "Definitivo\n(LGBM+todo)"]
mae = [8.19, 7.06, 4.40, 3.17, 3.15, 2.89, 2.80]
colors = ["#adb5bd","#868e96","#e8952f","#f08c00","#4dabf7","#3b8fd6","#2f6bd6"]

fig, ax = plt.subplots(1, 2, figsize=(16, 6))
b = ax[0].barh(range(len(modelos)), mae, color=colors)
ax[0].set_yticks(range(len(modelos))); ax[0].set_yticklabels(modelos)
ax[0].invert_yaxis(); ax[0].set_xlabel("MAE (días)")
ax[0].set_title("Progresión de modelos — predicción del tiempo de cosecha")
for i,v in enumerate(mae): ax[0].text(v+0.1, i, f"{v:.2f}", va="center", fontsize=10)
ax[0].axvline(2.80, color="#2f6bd6", ls=":", alpha=0.6)
ax[0].grid(alpha=0.3, axis="x")

# validación: agrupada vs temporal
esq = ["CV agrupada\n(por cosecha)", "Temporal\n80% train", "Temporal\n70% train", "Temporal\n60% train"]
maev = [2.80, 3.68, 4.09, 5.17]
ax[1].bar(esq, maev, color=["#2f6bd6","#f08c00","#e8952f","#e8590c"])
ax[1].set_ylabel("MAE (días)"); ax[1].set_title("Validación: agrupada (optimista) vs temporal (producción)")
for i,v in enumerate(maev): ax[1].text(i, v+0.05, f"{v:.2f}", ha="center", fontsize=10)
ax[1].grid(alpha=0.3, axis="y")
fig.tight_layout(); fig.savefig("resultados_consolidados.png", dpi=130)
print("Figura: resultados_consolidados.png")
