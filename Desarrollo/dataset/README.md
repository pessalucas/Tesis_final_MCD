# Dataset de Hidroponía — Albahaca

Este paquete convierte la planilla manual de registro de la compañía en un dataset
tabular limpio. Lo importante: **la mayor parte de la información operativa de la
planilla no está en el texto de las celdas, sino en su formato** (colores de relleno
y de borde). Un export a CSV la descarta. Este extractor la recupera.

El dataset entregado cubre **las dos hojas de tandas, filtrado a albahaca**:
**29.288 filas, 211 cosechas, de julio 2022 a septiembre 2026**, con **6.620 eventos
de manejo** recuperados de los colores.

## Contenido

| Archivo | Qué es |
|---|---|
| `extractor_hidroponia.py` | El extractor. Lee el `.xlsx` y produce el CSV. |
| `dataset_albahaca.csv` | El dataset: las dos hojas, sólo albahaca. |
| `README.md` | Este documento. |

## Cómo correrlo

```bash
pip install openpyxl

# lo que está entregado acá: las dos hojas, sólo albahaca
python extractor_hidroponia.py "Copia de Seguimiento de Parámetros.xlsx" \
    --cultivo Alb \
    --salida dataset_albahaca.csv

# todo, sin filtrar, con resumen por cosecha en pantalla
python extractor_hidroponia.py "Copia de Seguimiento de Parámetros.xlsx" --reporte

# una sola hoja
python extractor_hidroponia.py "Copia de Seguimiento de Parámetros.xlsx" \
    --hoja "Tandas Hidroponia 2"
```

`--cultivo` filtra por substring sin distinguir mayúsculas, así que `--cultivo Alb`
se queda con `Alb`, `Esquejes Alb`, `Alb + Lech` y `Alb + Chacras`, y descarta
rúcula, berro, cilantro y perejil.

Única dependencia: `openpyxl`. **Tiene que ser un `.xlsx`, no un CSV** — el CSV pierde
los colores, que es justamente lo que buscamos.

---

## Cómo está armada la planilla original

La planilla se llena a mano y guarda **tres cosas distintas en cada celda**:

1. **El valor** → pH y PPM medidos ese día.
2. **El color de relleno** → a qué tanque está conectada esa tanda.
3. **El color del borde inferior** → qué evento de manejo ocurrió ese día.

Además el layout es "en escalera": cada cosecha nueva se anota más abajo (fila = fecha
de inicio) y más a la derecha (columnas nuevas). Concretamente:

- **Columna A**: día de la semana. **Columna B**: fecha.
- **De la columna C en adelante**: pares de columnas `(pH, PPM)`. Cada par es una **tanda**.
- Una **cosecha** ocupa varios pares (típicamente 4 a 6), porque la alimentan varios
  tanques en simultáneo. **No siempre arrancan todos el mismo día**: en `T120` dos tandas
  empiezan en la fila 898 y las otras dos tres días después, en la 901.
- La **etiqueta** (`T202 Alb`) se escribe **una fila arriba** del primer día de datos,
  repetida sobre cada par.
- El string `-` significa **dato faltante** → se persiste como `null`.

### Ejemplo trabajado: cosecha T202

La etiqueta `T202 Alb` está en la fila 519, repetida en las columnas AEE, AEG, AEI,
AEK, AEM y AEO. Son **6 tandas**. El primer día de datos es la fila 520 → **05/07/26**.

Las seis arrancan sin tanque asignado (etapa pre-tanque, `ID_TANQUE = null`). Después:

| Tanda | Columna | Entra al tanque | Tanque |
|---|---|---|---|
| 1 | AEE | 24/07 | 2 |
| 2 | AEG | 24/07 | 2 |
| 3 | AEI | 24/07 | 1 |
| 4 | AEK | 24/07 | 1 |
| 5 | AEM | 28/07 | 2 |
| 6 | AEO | 28/07 | 2 |

Las tandas 1 y 2 cierran el 19/08 con borde bordó (cosecha); la 3 el 21/08; las
tandas 4, 5 y 6 llegan al final del registro sin cerrar.

---

## Diccionario de colores

Salió de los bloques de leyenda que están **escondidos a la derecha de los datos**
en la propia planilla (hoja 1: celda `RK956`; hoja 2: celda `AER573`).

### Relleno sólido → a qué tanque está conectada la tanda

| Color | Hex | Tanque |
|---|---|---|
| Gris claro | `D9D9D9` | 1 (izq) |
| Gris medio | `999999` | 2 (der) |
| Azul | `6FA8DC` | 3 |

> **Ojo con el tanque 2.** La leyenda dice que es el gris oscuro `666666`, pero ese
> color no se usa nunca en los datos — el que se usa es el gris medio `999999`.
> El extractor mapea los dos al tanque 2.

### Borde inferior de color → evento de manejo de ese día

| Color | Hex | Columna en el dataset |
|---|---|---|
| Amarillo | `FFFF00` | `CAMBIO_80_TANQUE` (recambio del 80% del tanque) |
| Naranja | `FF9900` | `ESTEN_80` |
| Violeta | `9900FF` | `AGREGADO_SALES` |
| Magenta | `FF00FF` | `CAMBIO_TANQUE` |
| Bordó | `980000` | `COSECHA_FIN` |
| Verde | `00FF00` | `BOVEL` (Bovel / Metarhizium) |
| Cyan | `00FFFF` | `G_A_F1` (germinación → fase 1) |

El borde negro es sólo la grilla de la planilla y se ignora.

### `TRASPASO_TANQUE` vs `CAMBIO_TANQUE`

Son cosas distintas y conviene no confundirlas:

- **`TRASPASO_TANQUE`** es *derivado*: vale 1 el día en que **cambia el color de relleno**,
  o sea el día en que esa tanda efectivamente pasó a estar en otro tanque.
- **`CAMBIO_TANQUE`** es el **borde magenta**: marca que ese día se hizo la maniobra.

No siempre coinciden. En T202 el magenta del 28/07 aparece sobre las seis tandas, pero
sólo las tandas 5 y 6 cambian de tanque ese día — las otras cuatro ya estaban en el suyo
desde el 24/07.

---

## Esquema del dataset

Grano: **una fila por `ID_COSECHA` × `ID_TANDA` × `FECHA`**.
La clave `(ID_COSECHA, ID_TANDA)` identifica un par de columnas de la planilla.
En el dataset entregado **no hay claves duplicadas**.

| Columna | Tipo | Descripción |
|---|---|---|
| `ID_COSECHA` | texto | Identificador de la cosecha: `T202`, `T174A`, `ESQUEJESALB1`. Ver "Series". |
| `ID_TANDA` | entero | Número de par de columnas dentro de la cosecha, de izquierda a derecha (1, 2, 3…). |
| `ID_TANQUE` | entero / null | Tanque que alimenta la tanda ese día. `null` = etapa pre-tanque. |
| `FECHA` | ISO `YYYY-MM-DD` | Fecha del registro. |
| `PH` | decimal / null | pH medido. `null` si en la planilla figuraba `-`. |
| `PPM` | decimal / null | Conductividad. **Ver la nota de unidades.** |
| `UNIDAD` | texto | Escala en que está `PPM`: `ppm` (hoja 1) o `mS/cm` (hoja 2). |
| `ORDEN_CORREGIDO` | 0/1 | 1 si en esa fila el operador cargó pH y PPM invertidos y el extractor los dio vuelta. |
| `TRASPASO_TANQUE` | 0/1 | 1 el día en que la tanda cambia de tanque (derivado del relleno). |
| `AGREGADO_SALES` | 0/1 | Borde violeta. |
| `CAMBIO_TANQUE` | 0/1 | Borde magenta. |
| `CAMBIO_80_TANQUE` | 0/1 | Borde amarillo. |
| `BOVEL` | 0/1 | Borde verde. |
| `ESTEN_80` | 0/1 | Borde naranja. |
| `COSECHA_FIN` | 0/1 | Borde bordó: cierre del ciclo de esa tanda. |
| `G_A_F1` | 0/1 | Borde cyan: pasaje de germinación a fase 1. |
| `CULTIVO` | texto | `Alb`, `Esquejes Alb`, `Alb + Lech`, `Alb + Chacras`. |
| `FRACCION` | texto | `1` = torre completa, `1/2` = media torre. |
| `DIA_DESDE_INICIO` | entero | Día 1, 2, 3… desde el arranque de esa tanda. |
| `HOJA` | texto | Hoja de origen. |
| `COLUMNA_XLSX` | texto | Columna original en la planilla (ej. `AEE`). Para auditar. |
| `FILA_XLSX` | entero | Fila original en la planilla. Para auditar. |

`COLUMNA_XLSX` + `FILA_XLSX` permiten volver a la celda exacta de la planilla desde
cualquier fila del dataset. Si un número parece raro, se puede ir a mirarlo.

### Series de `ID_COSECHA`

| Prefijo | Qué es |
|---|---|
| `T<n>` | Tanda numerada. `T1` a `T208`. Faltan `T11` y `T16` porque son de rúcula. |
| `T<n><letra>` | Sub-tanda que arranca unos días después de la principal. Ej: `T174A`. |
| `ESQUEJESALB<n>` | Tandas de esquejes de albahaca, sin número en la planilla: se numeran por orden de aparición. |
| `TMIXALB<n>` | La tanda rotulada `Tmix - Alb`. |

---

## Cifras del dataset

- **29.288 filas**, **211 cosechas**, sin claves duplicadas.
- Rango de fechas: **03/07/2022 → 04/09/2026**.
- Serie completa `T1` a `T208` (sin `T11` ni `T16`, que son rúcula).
- **6.620 eventos** recuperados de los bordes de color — información que no existía
  como dato en ningún lado:

| Evento | Cantidad |
|---|---|
| `ESTEN_80` | 2.466 |
| `CAMBIO_TANQUE` | 879 |
| `CAMBIO_80_TANQUE` | 831 |
| `AGREGADO_SALES` | 687 |
| `COSECHA_FIN` | 575 |
| `TRASPASO_TANQUE` | 493 |
| `BOVEL` | 421 |
| `G_A_F1` | 268 |

- Escalas: **9.542 filas en ppm** (hoja 1, jul-2022 a mar-2025) y **19.746 en mS/cm**
  (hoja 2, feb-2025 en adelante).
- Rangos observados: pH entre 2,66 y 7,10; PPM entre 4,2 y 2250 en ppm, y entre
  0,28 y 5,70 en mS/cm.

---

## Advertencias para quien use el dataset

**1. `PPM` viene en dos escalas distintas. Sin convertir.**
La hoja 1 registra ppm de verdad (50 a 2250); la hoja 2 registra **conductividad en
mS/cm** (0,3 a 5,7) aunque la columna se llame igual. El extractor detecta la escala
par por par y la deja en la columna `UNIDAD`, **sin tocar los valores**. Antes de
promediar, graficar o entrenar cualquier cosa sobre `PPM`, **hay que normalizar**. El
factor típico es `ppm = mS/cm × 500` o `× 640` según el medidor: confirmar cuál usa la
empresa. Filtrar por `UNIDAD` es la forma segura de trabajar con una sola escala.

**2. 54 filas tenían pH y PPM invertidos en el origen.**
En la escala ppm los dos valores son inconfundibles, así que cuando aparecen al revés
(ej. `CI284` dice `6,9 | 650` cuando toda la columna es `650 | 6,9`) el extractor los
da vuelta y marca la fila con `ORDEN_CORREGIDO = 1`. En la escala mS/cm los dos números
son chicos y una inversión de ese tipo **no es detectable**: puede haber casos sin
corregir en la hoja 2.

**3. Queda un valor de pH imposible sin corregir.**
`T59`, tanda 1, 15/12/2023 (celda `FC534`): tiene `1620` en la columna de pH y `1300`
en la de PPM. Los dos son de magnitud ppm, así que no se puede inferir cuál era el pH.
Se dejó tal cual para no inventar dato. **Conviene arreglarlo en la planilla.**

**4. Las dos hojas se solapan y el solapamiento ya está resuelto.**
`T128` a `T133` quedaron anotadas en las dos hojas (la hoja 1 sigue hasta el 07/03/2025
y la hoja 2 arranca el 02/02/2025). El extractor detecta la repetición, se queda con la
hoja que tiene la cosecha más completa —la 2 en los seis casos— y descarta las 514 filas
parciales de la hoja 1, informándolo por pantalla.

**5. Tandas duplicadas dentro de una misma cosecha.**
Cuando dos tandas comparten tanque suelen tener valores idénticos, porque se mide el
tanque y no la torre. En T202 las tandas 1 y 2 son idénticas. Si se promedia sin cuidado,
esas mediciones pesan doble. Para análisis a nivel tanque, deduplicar por
`(ID_COSECHA, ID_TANQUE, FECHA)`.

**6. Las tandas de una cosecha no cierran todas el mismo día.**
Cada par de columnas tiene su propio `COSECHA_FIN`. Algunas llegan al final del registro
sin bordó: son ciclos todavía abiertos, no datos faltantes.

**7. Hay tandas sin tanque asignado en toda su vida.**
Las tandas 5 y 6 de `T206`, por ejemplo, no tienen relleno en ninguna fila. Puede ser que
falte pintarlas en la planilla. Quedan con `ID_TANQUE = null` siempre.

**8. El dataset es un snapshot.**
Se generó de una descarga puntual del Google Sheet. Para actualizarlo hay que volver a
bajar el `.xlsx` (Archivo → Descargar → Microsoft Excel) y correr el extractor de nuevo.
No hay conexión en vivo.

---

## Pendiente

- Definir el factor de conversión de mS/cm a ppm y normalizar la columna `PPM`.
- Corregir `FC534` en la planilla.
- Evaluar si `CAMBIO_TANQUE` debería renombrarse a algo como `MANIOBRA_TANQUE`, para que
  no se confunda con `TRASPASO_TANQUE`.
