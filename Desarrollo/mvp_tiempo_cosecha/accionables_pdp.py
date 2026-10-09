"""Objetivo 4 — factores accionables: dependencia parcial (PDP) + recomendaciones.
Completa lo que faltaba del obj 4: cómo cambia la predicción al mover las variables que el
operador controla (pH, PPM, eventos de manejo), con discusión causal.

Correr: uv run accionables_pdp.py
"""
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor

RNG = 42
L = pd.read_parquet("landmarks_full.parquet")
FEATS = [c for c in L.columns if c not in ["ID_COSECHA", "ID_TANDA", "D"]]
y = L["D"].to_numpy(float)
BEST = dict(loss_function="MAE", depth=6, learning_rate=0.03, l2_leaf_reg=1, iterations=500,
            subsample=0.7, random_state=RNG, verbose=0, thread_count=-1, allow_writing_files=False)
model = CatBoostRegressor(**BEST).fit(L[FEATS], y)

# variables controlables por el operador
ACC = ["ppm_mean", "ppm_min", "ph_mean", "ph_std", "n_BOVEL", "n_AGREGADO_SALES",
       "dias_desde_AGREGADO_SALES"]
ACC = [a for a in ACC if a in FEATS]
Xs = L[FEATS].sample(min(3000, len(L)), random_state=RNG).copy()

def pdp(feat, grid):
    base = Xs.copy(); out = []
    for v in grid:
        base[feat] = v
        out.append(model.predict(base).mean())
    return np.array(out)

print("=== Dependencia parcial (PDP) — efecto en la DURACIÓN predicha (días) ===")
fig, axes = plt.subplots(2, 4, figsize=(18, 8)); axes = axes.ravel()
resumen = []
for i, f in enumerate(ACC):
    lo, hi = np.nanpercentile(L[f], [5, 95])
    grid = np.linspace(lo, hi, 30)
    pd_vals = pdp(f, grid)
    efecto = pd_vals[-1] - pd_vals[0]  # cambio de duración de p5 a p95
    resumen.append((f, efecto))
    axes[i].plot(grid, pd_vals, color="#2f7de8", lw=2)
    axes[i].set_title(f"{f}  (Δ={efecto:+.1f} d)"); axes[i].set_xlabel(f); axes[i].set_ylabel("duración pred (d)")
    axes[i].grid(alpha=0.3)
    print(f"  {f:28s} de p5 a p95 -> duración {pd_vals[0]:.1f} a {pd_vals[-1]:.1f}  (Δ {efecto:+.1f} d)")
for j in range(len(ACC), len(axes)): axes[j].axis("off")
plt.suptitle("Dependencia parcial de las variables accionables (campeón CatBoost)", fontsize=13)
plt.tight_layout(); plt.savefig("accionables_pdp.png", dpi=130)
print("\nGuardado: accionables_pdp.png")

print("\n=== Magnitud del efecto accionable (|Δ duración| de p5 a p95) ===")
for f, e in sorted(resumen, key=lambda x: -abs(x[1])):
    print(f"  {f:28s} {e:+.1f} d")
pd.DataFrame(resumen, columns=["feature", "delta_dias_p5_p95"]).to_csv("accionables_pdp.csv", index=False)
