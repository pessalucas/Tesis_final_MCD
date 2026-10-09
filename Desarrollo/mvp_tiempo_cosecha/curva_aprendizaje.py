"""Curva de aprendizaje — MAE vs nº de cosechas de entrenamiento.
Evidencia central de la tesis: el cuello de botella es el tamaño de muestra (cosechas),
no el modelo. Test FIJO temporal (último año); se varía cuántas cosechas del pasado se usan
para entrenar, con repeticiones para estimar la varianza. Si la curva sigue bajando al final,
más cosechas mejorarían el desempeño.

Correr: uv run curva_aprendizaje.py
"""
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from lightgbm import LGBMRegressor

RNG = 42; CUT = pd.Timestamp("2025-09-26")
L = pd.read_parquet("landmarks_full.parquet")
raw = pd.read_csv("../dataset/dataset_albahaca_1.csv", parse_dates=["FECHA"])
starts = raw.groupby("ID_COSECHA")["FECHA"].min().rename("start")
L = L.merge(starts, on="ID_COSECHA", how="left")
FEATS = [c for c in L.columns if c not in ["ID_COSECHA", "ID_TANDA", "D", "start"]]

pool = L[L["start"] < CUT]; test = L[L["start"] >= CUT]
pool_cos = pool["ID_COSECHA"].unique()
Xte, yte = test[FEATS], test["D"].to_numpy(float)
print(f"Cosechas disponibles para entrenar: {len(pool_cos)} | test fijo: {test['ID_COSECHA'].nunique()} cosechas")

LGB = dict(objective="regression_l1", n_estimators=400, learning_rate=0.05, num_leaves=31,
           min_child_samples=20, reg_lambda=3, subsample=0.8, subsample_freq=1,
           colsample_bytree=0.8, n_jobs=-1, verbose=-1, random_state=RNG)
sizes = [20, 40, 60, 80, 100, 120, len(pool_cos)]
R = 10
rng = np.random.default_rng(RNG)
rows = []
for n in sizes:
    maes = []
    reps = 1 if n >= len(pool_cos) else R
    for rep in range(reps):
        sub = pool_cos if n >= len(pool_cos) else rng.choice(pool_cos, size=n, replace=False)
        tr = pool[pool["ID_COSECHA"].isin(sub)]
        m = LGBMRegressor(**LGB).fit(tr[FEATS], tr["D"].to_numpy(float))
        maes.append(mean_absolute_error(yte, m.predict(Xte)))
    rows.append((n, float(np.mean(maes)), float(np.std(maes))))
    print(f"  n_cosechas={n:3d}: MAE test = {np.mean(maes):.3f} ± {np.std(maes):.3f}")

df = pd.DataFrame(rows, columns=["n_cosechas", "mae", "sd"])
df.to_csv("curva_aprendizaje.csv", index=False)
plt.figure(figsize=(7.5, 5))
plt.errorbar(df["n_cosechas"], df["mae"], yerr=df["sd"], marker="o", color="#2f7de8", capsize=4)
plt.xlabel("Nº de cosechas de entrenamiento"); plt.ylabel("MAE en test temporal (días)")
plt.title("Curva de aprendizaje — el tamaño de muestra manda")
plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig("curva_aprendizaje.png", dpi=130)

d = df.sort_values("n_cosechas").reset_index(drop=True)
slope = (d["mae"].iloc[-1] - d["mae"].iloc[-2]) / (d["n_cosechas"].iloc[-1] - d["n_cosechas"].iloc[-2])
print(f"\nMejora total {d['mae'].iloc[0]:.2f} -> {d['mae'].iloc[-1]:.2f} d")
print(f"Pendiente al final: {slope:.4f} d/cosecha -> "
      f"{'SIGUE BAJANDO: más cosechas ayudarían' if slope < -0.002 else 'se aplanó'}")
print("Guardado: curva_aprendizaje.png, curva_aprendizaje.csv")
