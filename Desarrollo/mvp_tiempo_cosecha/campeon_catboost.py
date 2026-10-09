"""Campeón = CatBoost (ganador del challenge de tecnologías).

Re-corre sobre CatBoost las piezas del eje:
  1. Métricas del campeón: OOF GroupKFold por cosecha (MAE/MedAE/RMSE) + test temporal.
  2. SHAP nativo de CatBoost -> factores accionables (importancia por |SHAP|).
  3. Intervalos conformal (CQR) con cuantiles de CatBoost, split-conformal por cosecha.

Hiperparámetros: los mejores del challenge (benchmark.py):
  depth=6, learning_rate=0.03, l2_leaf_reg=1, iterations=500, subsample=0.7.

Correr: uv run campeon_catboost.py
"""
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, median_absolute_error, mean_squared_error
from catboost import CatBoostRegressor, Pool

RNG = 42
CUT = pd.Timestamp("2025-09-26")
L = pd.read_parquet("landmarks_full.parquet")
raw = pd.read_csv("../dataset/dataset_albahaca_1.csv", parse_dates=["FECHA"])
starts = raw.groupby("ID_COSECHA")["FECHA"].min().rename("start")
L = L.merge(starts, on="ID_COSECHA", how="left")
FEATS = [c for c in L.columns if c not in ["ID_COSECHA", "ID_TANDA", "D", "start"]]
y = L["D"].to_numpy(float)
groups = L["ID_COSECHA"].to_numpy()

BEST = dict(loss_function="MAE", depth=6, learning_rate=0.03, l2_leaf_reg=1,
            iterations=500, subsample=0.7, random_state=RNG, verbose=0,
            thread_count=-1, allow_writing_files=False)

def metrics(yt, p):
    return mean_absolute_error(yt, p), median_absolute_error(yt, p), np.sqrt(mean_squared_error(yt, p))

# ---------- 1. métricas del campeón ----------
print("=" * 60)
print("CAMPEÓN CatBoost — métricas")
print("=" * 60)
oof = np.zeros(len(L))
for tr, te in GroupKFold(5).split(L, y, groups):
    oof[te] = CatBoostRegressor(**BEST).fit(L.iloc[tr][FEATS], y[tr]).predict(L.iloc[te][FEATS])
mae, medae, rmse = metrics(y, oof)
print(f"CV agrupada (GroupKFold): MAE {mae:.3f} | MedAE {medae:.3f} | RMSE {rmse:.3f}")
np.save("oof_catboost.npy", oof)

trm, tem = (L["start"] < CUT).values, (L["start"] >= CUT).values
pt = CatBoostRegressor(**BEST).fit(L[trm][FEATS], y[trm]).predict(L[tem][FEATS])
mae_t, medae_t, rmse_t = metrics(y[tem], pt)
print(f"Test temporal (último año, {tem.sum()} landmarks): "
      f"MAE {mae_t:.3f} | MedAE {medae_t:.3f} | RMSE {rmse_t:.3f}")

# ---------- 2. SHAP nativo ----------
print("\n" + "=" * 60)
print("SHAP — factores más influyentes (|SHAP| medio)")
print("=" * 60)
model_full = CatBoostRegressor(**BEST).fit(L[FEATS], y)
sv = model_full.get_feature_importance(Pool(L[FEATS], y), type="ShapValues")[:, :-1]
imp = np.abs(sv).mean(0)
order = np.argsort(imp)[::-1]
top = [(FEATS[i], imp[i]) for i in order[:20]]
for f, v in top:
    print(f"  {f:28s} {v:.3f}")
pd.DataFrame({"feature": [FEATS[i] for i in order], "shap_abs_mean": imp[order]}) \
    .to_csv("shap_catboost.csv", index=False)

plt.figure(figsize=(8, 6))
names = [t[0] for t in top][::-1]
vals = [t[1] for t in top][::-1]
plt.barh(names, vals, color="#2f7de8")
plt.xlabel("|SHAP| medio (impacto en la predicción, días)")
plt.title("CatBoost — top 20 features por importancia SHAP")
plt.tight_layout()
plt.savefig("shap_catboost.png", dpi=130)
print("Guardado: shap_catboost.csv, shap_catboost.png")

# ---------- 3. Conformal (CQR) ----------
print("\n" + "=" * 60)
print("Intervalos conformal (CQR) con CatBoost — objetivo 80%")
print("=" * 60)
yr = (L["D"] - L["dia_actual"]).to_numpy(float)  # días restantes
ALPHA = 0.10

def q(a):
    return CatBoostRegressor(loss_function=f"Quantile:alpha={a}", depth=6, learning_rate=0.05,
                             iterations=400, l2_leaf_reg=3, subsample=0.8, random_state=RNG,
                             verbose=0, thread_count=-1, allow_writing_files=False)

rng = np.random.default_rng(RNG)
lo_n = np.zeros(len(L)); hi_n = np.zeros(len(L)); lo_c = np.zeros(len(L)); hi_c = np.zeros(len(L))
for tr_all, te in GroupKFold(5).split(L, yr, groups):
    cos = pd.unique(groups[tr_all]); rng.shuffle(cos)
    cal_cos = set(cos[:int(len(cos) * 0.3)])
    cal = tr_all[np.isin(groups[tr_all], list(cal_cos))]
    tr = tr_all[~np.isin(groups[tr_all], list(cal_cos))]
    mlo = q(ALPHA).fit(L.iloc[tr][FEATS], yr[tr])
    mhi = q(1 - ALPHA).fit(L.iloc[tr][FEATS], yr[tr])
    loc, hic = mlo.predict(L.iloc[cal][FEATS]), mhi.predict(L.iloc[cal][FEATS])
    s = np.maximum(loc - yr[cal], yr[cal] - hic)
    k = min(int(np.ceil((len(s) + 1) * (1 - 2 * ALPHA))), len(s))
    E = np.sort(s)[k - 1]
    lot, hit = mlo.predict(L.iloc[te][FEATS]), mhi.predict(L.iloc[te][FEATS])
    lo_n[te], hi_n[te] = lot, hit
    lo_c[te], hi_c[te] = lot - E, hit + E

def cov(lo, hi): return np.mean((yr >= lo) & (yr <= hi)) * 100
def wid(lo, hi): return np.mean(hi - lo)
print(f"  naive (cuantiles): cobertura {cov(lo_n, hi_n):.1f}% | ancho {wid(lo_n, hi_n):.1f} d")
print(f"  CQR   (conformal): cobertura {cov(lo_c, hi_c):.1f}% | ancho {wid(lo_c, hi_c):.1f} d  (obj. 80%)")
print("\n  Cobertura/ancho CQR por etapa del ciclo:")
for lo, hi in [(3, 10), (11, 20), (21, 30), (31, 45), (46, 100)]:
    m = ((L["dia_actual"] >= lo) & (L["dia_actual"] <= hi)).values
    if m.sum():
        print(f"    día {lo:>2}-{hi:<3}: cob {np.mean((yr[m] >= lo_c[m]) & (yr[m] <= hi_c[m])) * 100:5.1f}% "
              f"| ancho {np.mean(hi_c[m] - lo_c[m]):.1f} d")
print("\nLISTO.")
