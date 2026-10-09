"""Sonda: ¿los eventos aportan señal de duración o no?
Investiga por qué la permutation importance de varios eventos da ~0.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error

RNG = 42
EVENTOS = ["TRASPASO_TANQUE", "AGREGADO_SALES", "CAMBIO_TANQUE", "CAMBIO_80_TANQUE",
           "BOVEL", "ESTEN_80", "G_A_F1"]
L = pd.read_parquet("landmarks.parquet")
y = L["D"].to_numpy(float)
groups = L["ID_COSECHA"].to_numpy()
gkf = GroupKFold(n_splits=5)

def cv_mae(cols):
    oof = np.zeros(len(L))
    for tr, te in gkf.split(L, y, groups):
        m = HistGradientBoostingRegressor(max_depth=4, max_iter=300, learning_rate=0.05,
                                          min_samples_leaf=40, l2_regularization=1.0,
                                          random_state=RNG).fit(L.iloc[tr][cols], y[tr])
        oof[te] = m.predict(L.iloc[te][cols])
    return mean_absolute_error(y, oof)

n_cols = [f"n_{e}" for e in EVENTOS]
dd_cols = [f"dias_desde_{e}" for e in EVENTOS]
event_cols = n_cols + dd_cols

# ---------------------------------------------------------------------------
print("======  PRUEBA 1: ¿tienen señal de duración los eventos, SOLOS?  ======")
dummy = mean_absolute_error(y, np.full(len(y), y.mean()))
print(f"  Dummy (constante)                 MAE = {dummy:5.2f} d")
print(f"  Solo dia_actual                   MAE = {cv_mae(['dia_actual']):5.2f} d")
print(f"  Solo eventos (conteo + recencia)  MAE = {cv_mae(event_cols):5.2f} d")
print(f"  Eventos + dia_actual              MAE = {cv_mae(event_cols + ['dia_actual']):5.2f} d")
print(f"  Solo mes (estacion)               MAE = {cv_mae(['mes_inicio','mes_sin','mes_cos']):5.2f} d")

# ---------------------------------------------------------------------------
print("\n======  PRUEBA 2: ¿son redundantes con 'dia_actual'?  ======")
print("  (corr de cada feature de evento con el dia transcurrido)\n")
for c in n_cols + dd_cols:
    r = np.corrcoef(L[c].fillna(L[c].median()), L["dia_actual"])[0, 1]
    print(f"    {c:28s} corr(dia_actual) = {r:+.2f}")

# ---------------------------------------------------------------------------
print("\n======  PRUEBA 3: conteo CRUDO vs TASA por dia  ======")
print("  Los ciclos largos acumulan mas eventos solo por durar mas.")
print("  ¿Sirve mejor la tasa (eventos/dia) que el conteo crudo?\n")
Lr = L.copy()
for e in EVENTOS:
    Lr[f"tasa_{e}"] = Lr[f"n_{e}"] / Lr["dia_actual"].clip(lower=1)
tasa_cols = [f"tasa_{e}" for e in EVENTOS]
# hay que recomputar cv con Lr; adapto cv_mae
def cv_mae_df(df, cols):
    oof = np.zeros(len(df))
    for tr, te in gkf.split(df, y, groups):
        m = HistGradientBoostingRegressor(max_depth=4, max_iter=300, learning_rate=0.05,
                                          min_samples_leaf=40, l2_regularization=1.0,
                                          random_state=RNG).fit(df.iloc[tr][cols], y[tr])
        oof[te] = m.predict(df.iloc[te][cols])
    return mean_absolute_error(y, oof)
print(f"  Solo conteos (n_)                 MAE = {cv_mae_df(Lr, n_cols):5.2f} d")
print(f"  Solo tasas (eventos/dia)          MAE = {cv_mae_df(Lr, tasa_cols):5.2f} d")
print(f"  Solo recencia (dias_desde_)       MAE = {cv_mae_df(Lr, dd_cols):5.2f} d")

# ---------------------------------------------------------------------------
print("\n======  PRUEBA 4: ¿aportan eventos SOBRE estacion + dia?  ======")
base = ["mes_inicio","mes_sin","mes_cos","dia_actual"]
print(f"  base (mes + dia_actual)                       MAE = {cv_mae(base):5.2f} d")
print(f"  base + eventos (conteo+recencia)              MAE = {cv_mae(base + event_cols):5.2f} d")
print(f"  base + eventos + tasas                        "
      f"MAE = {cv_mae_df(Lr, base + event_cols + tasa_cols):5.2f} d")
