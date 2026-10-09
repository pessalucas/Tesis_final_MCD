# EDA — Predicción del tiempo de cosecha (albahaca)

Tandas completas: 565 | cosechas: 184
Rango temporal: 2022-07-03 -> 2026-07-05

## 1. Distribución del target (duración total, días)
- media 47.0 | mediana 46.0 | sd 9.8 | min 28 | max 92
- rango intercuartil: [39, 54] | asimetría (skew) 0.53 | curtosis 0.30
- Interpretación: skew +0.53 => cola derecha (algunas cosechas muy largas).

## 2. Consistencia a lo largo del tiempo
Duración por año de inicio:
      count  median       mean        std
anio                                     
2022     16    51.0  52.687500   5.485967
2023     37    54.0  54.648649   9.022245
2024    100    39.0  43.640000  11.833457
2025    261    44.0  45.750958   9.202176
2026    151    46.0  48.913907   8.272799

Mediana de mediciones por tanda (densidad de registro) por año:
anio
2022    51.0
2023    54.0
2024    39.0
2025    44.0
2026    46.0

Faltantes (%) por año:
      ph_falt  ppm_falt
anio                   
2022     20.4      19.9
2023     22.9      22.9
2024     13.5      13.5
2025      0.4       0.4
2026      5.7       6.1

Unidad de PPM por hoja (el shift de instrumentación):
HOJA                 UNIDAD
Tandas Hidroponia 1  ppm        9432
Tandas Hidroponia 2  mS/cm     19746

## 3. Consistencia TRAIN vs TEST (último año)
TRAIN: 335 tandas | TEST: 230 tandas

               TRAIN     TEST
dur_media      47.90    45.70
dur_mediana    48.00    44.00
dur_sd         10.62     8.34
ppm_medio    1139.45  1211.02
ph_medio        6.12     5.80

Test KS de la duración (TRAIN vs TEST): D=0.280, p=0.000 => distribuciones DIFERENTES (hay shift)

## 4. Relación features <-> target (Spearman, nivel landmark)
Top 12 correlación NEGATIVA con la duración:
cl_fut14_dli_clim      -0.782
cl_fut7_dli_clim       -0.780
cl_fc_dli_next7        -0.776
cl_fut30_dli_clim      -0.772
cl_fc_dli_next7_anom   -0.765
cl_fut30_temp_clim     -0.759
cl_fut30_gdd_clim      -0.757
cl_dli_last14          -0.752
cl_dli_last7           -0.749
cl_dli_mean_ciclo      -0.742
cl_dli_min_ciclo       -0.741
cl_fc_dli_d1           -0.737

Top 12 correlación POSITIVA con la duración:
dia_actual             0.301
dias_desde_G_A_F1      0.309
mes_sin                0.321
n_BOVEL                0.390
cl_prec_acum_ciclo     0.429
cl_fc_cloud_next7      0.499
cl_cloud_mean_ciclo    0.503
hist_dur_reciente5     0.508
cl_ndias_frios         0.586
cl_rh_mean_ciclo       0.635
hist_dur_mismomes      0.655
cl_ndias_pocaluz       0.731


## Figuras
- eda_target.png (distribución + estacionalidad)
- eda_consistencia_temporal.png (drift por año, densidad, faltantes)
- eda_train_vs_test.png (comparación de distribuciones)
- eda_corr_target.png (correlaciones con el target)
- eda_scatter_target.png (dispersión de las 4 más correlacionadas)
