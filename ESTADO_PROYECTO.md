# Estado del proyecto — House Prices

Documento de continuidad. Resume dónde quedó el trabajo, qué decisiones se tomaron y por qué,
y cuál es el siguiente paso concreto.

**Última actualización:** 2026-11-04

---

## 1. Qué es esto

Predicción de precios de vivienda con el dataset *House Prices: Advanced Regression
Techniques* de Kaggle. Estructura generada con **Cookiecutter Data Science v2.3.0**.

- **Objetivo (target):** `SalePrice`
- **Métrica oficial de la competencia:** RMSE sobre `log(SalePrice)`
- **Datos:** 1460 filas de entrenamiento × 81 columnas; 1459 de test
- **Estado:** pipeline completo terminado, de datos crudos a script de inferencia con pruebas

**Resultado actual:** ElasticNet (`alpha=0.001`, `l1_ratio=0.5`) con **20 042 USD** de RMSE
real, **4.01 % del rango** y 0.1401 de RMSE logarítmico, medido sobre 292 casas reservadas.

---

## 2. Entorno — leer antes de ejecutar nada

El entorno del proyecto es el conda env **`env_house_pricing`** (Python 3.11, conda-forge).

```bash
conda activate env_house_pricing
pip install -r requirements.txt
```

### Trampa conocida nº 1: hay que ejecutar con el entorno activado

**No usar `ciencia_datos_python_2026`.** Y tampoco invocar directamente el `python.exe` de
`env_house_pricing` sin activar: cualquier multiplicación de matrices 2-D mata el proceso con
un crash nativo (`0xc06d007f`, DLL no encontrada) porque `Library\bin` —donde viven
`libblas.dll` y MKL— no entra al PATH de búsqueda de DLLs.

El síntoma es un "Kernel died" sin excepción de Python. Si algo revienta sin traza, sospechar
del PATH de DLLs antes que del código. Desde terminal, la invocación segura es:

```bash
conda run -n env_house_pricing --no-capture-output python script.py
```

`conda run` **no acepta `python -c` con saltos de línea**: falla con `NotImplementedError`.
Para código de varias líneas hay que escribirlo primero a un archivo.

### Trampa conocida nº 2: VS Code apunta al entorno viejo

`.vscode/settings.json` todavía apunta a `ciencia_datos_python_2026`. Por eso VS Code marca
todos los paquetes de `requirements.txt` como "not installed". **Pendiente de cambiar a
`env_house_pricing`.** Es solo ruido del editor: la ejecución real no se ve afectada.

### Trampa conocida nº 3: refrescar notebooks en el editor

Los notebooks se ejecutaron desde terminal con `nbconvert --inplace`, que reescribe el archivo
por debajo del editor. VS Code no siempre detecta ese cambio externo: si ves una versión
vieja, cierra y reabre la pestaña.

---

## 3. Qué está hecho

Seis fases, todas ejecutadas sin errores.

### `notebooks/1.0-dg-analisis-exploratorio.ipynb` — 33 celdas

| Sección | Contenido |
|---|---|
| 1 | Vistazo general: 36 numéricas, 43 categóricas, 0 duplicados |
| 2 | `SalePrice`: sesgo 1.88 → **0.12** con `log1p` |
| 3 | Faltantes: 34 columnas, 16 de ellas **solo en test** |
| 4 | Correlaciones, matriz de redundancia, outliers de `GrLivArea` |
| 4.1 | Separación numéricas: **15 discretas / 21 continuas** (regla: `nunique <= 15`) |
| 4.2 | Scatter de las 21 continuas, con `r` en el título y línea de tendencia |
| 4.3 | Boxplot + stripplot de las 15 discretas |
| 5 | Categóricas: boxplot de `OverallQual`, precio por barrio, casi constantes |
| 5.1 | Boxplot de las 43 categóricas con **η²** en el título |

### `notebooks/2.0-dg-limpieza-datos.ipynb`

Entrada `data/raw/` → salida `data/interim/{train,test}_limpio.csv`.

| Paso | Resultado |
|---|---|
| Faltantes al entrar | 7 829 train / 7 878 test |
| Outliers | 1460 → **1458** filas (Id 524 y 1299, ventas `Partial`); test intacto en 1459 |
| Faltantes al salir | **0 / 0** |
| Ordinales codificadas | 20 columnas, sin niveles huérfanos |
| Nominales descartadas | 6 (> 96 % de dominancia) |
| Forma final | train (1458, 76) · test (1459, 75) |

### `notebooks/3.0-dg-feature-engineering.ipynb` — 26 celdas

Produce el artefacto que sostiene todo lo demás: un **`Pipeline` de 13 pasos** con
transformadores de feature-engine, ajustado con train y aplicado igual a test.

| Etapa | Pasos |
|---|---|
| 1 · Imputación | `AddMissingIndicator`, `RellenarAnioGaraje`, estructurales cat y num, muestreo aleatorio, mediana, moda |
| 2 · Codificación | `OrdinalEncoder` con escalas escritas a mano; `OneHotEncoder` para nominales |
| 3 · Outliers | `Winsorizer` al percentil 1 y 99 |
| 4 · Transformación | `LogCpTransformer` y `YeoJohnsonTransformer` |
| 5 · Escalado | `StandardScaler` sobre las 20 numéricas |

**30 features crudas → 122 columnas.** Salidas: `data/processed/{train,test}_features.csv`,
`models/pipeline_features.joblib` y `data/processed/features_seleccionadas.yaml`.

### `notebooks/4.0-dg-modelado.ipynb` — 36 celdas, 9 figuras

Torneo de **8 modelos · 66 configuraciones · 10 pliegues = 660 ajustes** del pipeline
completo. Tiempo de ejecución: unos 7 minutos.

| Modelo | RMSE log CV | Desv. | Brecha | RMSE log val. | RMSE USD |
|---|---|---|---|---|---|
| Gradient Boosting | 0.1259 | 0.0136 | 0.0514 | 0.1438 | 20 282 |
| **ElasticNet (ganador)** | **0.1261** | **0.0092** | **0.0090** | **0.1401** | **20 042** |
| Lasso | 0.1265 | 0.0102 | 0.0112 | 0.1399 | 20 127 |
| Ridge | 0.1271 | 0.0105 | 0.0107 | 0.1425 | 20 371 |
| Random Forest | 0.1332 | 0.0170 | 0.0833 | 0.1538 | 22 240 |
| Hist Gradient Boosting | 0.1336 | 0.0186 | 0.0467 | 0.1538 | 22 384 |
| Regresión lineal | 0.1338 | 0.0142 | 0.0209 | 0.1445 | 21 228 |
| Red neuronal (MLP) | 0.1374 | 0.0146 | 0.0170 | 0.1447 | 21 503 |

Salidas: `models/modelo_ganador.joblib`, `models/modelo_produccion.joblib`,
`reports/torneo_modelos.csv`.

### `notebooks/5.0-dg-inferencia.ipynb` — 18 celdas, 2 figuras

De `data/raw/test.csv` crudo a `data/processed/submission.csv`. Sin un solo `fit`.
1459 predicciones entre 44 119 y 589 374 USD, ninguna fuera del rango visto en train.

### Fase 6 — Puesta a disposición del modelo

| Archivo | Contenido |
|---|---|
| `house_pricing/modeling/predict.py` | Biblioteca de inferencia, tipada y documentada |
| `scripts/predecir.py` | CLI de typer: `demo`, `archivo`, `ficha` |
| `tests/test_predict.py` | 29 pruebas, todas pasan |
| `scripts/logs/` | Un log por ejecución, con estampa de tiempo |

`make lint` limpio en los 10 archivos. El script reproduce el `submission.csv` del notebook
5.0 **byte a byte**.

### Documentación entregable

`reports/proceso.docx` (18 páginas) y `reports/prompts.docx` (12 páginas), generados con
`python-docx`.

---

## 4. Decisiones tomadas y su justificación

**Solo 2 de los 4 outliers de `GrLivArea` salen.** Los Id 524 y 1299 son calidad 10, enormes,
vendidos bajo 200 mil con `SaleCondition = Partial` (obra inconclusa): el precio no refleja el
inmueble. Los Id 692 y 1183 también son enormes pero cuestan ~750 mil, o sea coherentes, y
sostienen el extremo alto del rango. Eliminar los cuatro, como hacen muchos kernels de Kaggle,
tira información buena.

**`GarageYrBlt` no se rellena con 0.** Sin garaje, el año no es cero ni es un dato perdido: no
existe. Un 0 inventaría casas construidas en el año cero. Se rellena con `YearBuilt`
(correlación 0.83) más un indicador `SinGaraje`.

**Faltantes estructurales vs. reales.** En este dataset `NA` suele significar *ausencia de la
característica* (`PoolQC` nulo = sin piscina), no dato perdido. Imputar con la moda inventaría
piscinas. Se clasificaron las 34 columnas con faltantes en estructurales (→ `"None"` o `0`) y
reales (→ mediana o moda de train).

**η² como métrica para categóricas.** Pearson no aplica sin orden numérico. η² = SS_entre /
SS_total se lee como proporción de varianza del precio explicada por la variable. Su raíz η
está en la misma escala que |r| — de hecho, con dos niveles, η es exactamente |r|.

**El modelado parte de `data/raw`, no de `data/processed`.** `train_features.csv` salió de un
pipeline ajustado con las 1458 filas completas: su escalado, sus percentiles de recorte y sus
medianas ya vieron todas las filas. Usarlo para la validación cruzada contaminaría cada
pliegue. El notebook 4.0 carga `pipeline_features.joblib`, lo **clona** con `sklearn.base.clone`
y lo pone como primer paso de un `Pipeline` que termina en el estimador, de modo que las 13
transformaciones se re-ajustan dentro de cada pliegue. El costo es tiempo de cómputo.

**El ganador se elige con la regla de 1 error estándar, no con el mejor promedio.** Gradient
Boosting ganó el promedio de CV por 0.0002, el **1 % de su propia desviación entre pliegues**,
y a cambio sobreajusta 5.7 veces más y pierde en el conjunto reservado. La regla declara
empatados a los modelos dentro de un error estándar del mejor (4 de 8) y se queda con el de
menor brecha de sobreajuste. "Más simple" se define con una cantidad **medida** y comparable
entre familias, no con una opinión. Está escrito como código: si los números cambian, la
decisión cambia sola.

**Dos artefactos de modelo, no uno.** `modelo_ganador.joblib` (80 % de train) es **evidencia**:
es el objeto exacto que produjo los 20 042 USD, y mientras exista esa cifra es verificable.
`modelo_produccion.joblib` (100 %) es la **herramienta**. Los hiperparámetros del segundo se
heredan del torneo: volver a buscarlos sobre los datos completos usaría las filas de validación
para elegir.

**La lógica de inferencia vive en el paquete, no en el script.** `scripts/predecir.py` solo
traduce argumentos de consola a llamadas de `house_pricing.modeling.predict`. El servicio de
FastAPI debe importar esas mismas funciones, no una copia que se desincronice.

---

## 5. Reservas sobre la selección de features — resueltas en el notebook 3.0

Las tres reservas que estaban abiertas al 2026-10-29 se resolvieron re-ejecutando la selección
sobre los datos ya codificados:

- **`PoolArea` salió.** Su `r = 0.092` venía de 7 casas; 1453 de 1460 valen 0.
- **Los cuatro pares redundantes** (`GarageArea`/`GarageCars` 0.882, `GarageYrBlt`/`YearBuilt`
  0.826, `GrLivArea`/`TotRmsAbvGrd` 0.825, `1stFlrSF`/`TotalBsmtSF` 0.820) se conservaron: el
  Lasso y el ElasticNet del torneo se encargan de descontar la redundancia por su cuenta, y la
  importancia por permutación confirmó que no estorban.
- **Las 7 "categóricas" que ya eran numéricas** entraron a la selección con Pearson, como el
  resto. La partición en tres grupos de diez se rehízo sobre los datos limpios.

**La lista vigente es `data/processed/features_seleccionadas.yaml`**, no `pipeline_config.yaml`,
que quedó como registro histórico de la primera selección.

---

## 6. Plan de transformaciones por sesgo — implementado en el notebook 3.0

Se aplicó el criterio de tratar el sesgo **según su causa**, no según su magnitud:

| Causa | Variables | Tratamiento aplicado |
|---|---|---|
| Asimetría genuina, sin ceros | `GrLivArea`, `1stFlrSF` | `log1p` |
| Donde el logaritmo no aplica | `LotFrontage` | Yeo-Johnson |
| Años | `YearBuilt`, `YearRemodAdd`, `GarageYrBlt` | Sin transformar |
| Resto de continuas | — | Solo recorte de percentiles y escalado |

### Advertencia verificada sobre estos datos

Logaritmar a ciegas **destruye información**:

| Variable | % ceros | r original | r con log-log | Sesgo original | Sesgo tras log |
|---|---|---|---|---|---|
| `TotalBsmtSF` | 2.5 % | 0.614 | **0.373** | 1.52 | **−5.15** |
| `GarageArea` | 5.5 % | 0.623 | **0.455** | 0.18 | **−3.48** |

Mecanismo: los ceros se mapean a `log(1) = 0` y el resto cae cerca de `log(1000) ≈ 6.9`,
abriendo un abismo artificial entre "sin sótano" y "con sótano". Y el logaritmo tampoco
arregla la inflación de ceros: **ninguna transformación monótona elimina un pico de masa en un
solo valor**.

**El target sí se transforma:** `log1p(SalePrice)` lleva el sesgo de 1.88 a 0.12 y coincide con
la métrica oficial de Kaggle.

---

## 7. Hallazgos del torneo

**La elección del modelo importa mucho menos que tener buenas features.** Los ocho candidatos
caben entre 20 042 y 22 384 dólares. El predictor trivial —responder siempre la mediana— se
equivoca en 68 272. Pasar de "nada" a "cualquier modelo" vale unos 47 000 dólares de error;
elegir bien entre los ocho vale 2 300.

**Los ensambles son los que más memorizan, y no lo cobran.** Las tres brechas más grandes son
de árboles. Random Forest baja a 0.050 de error dentro del entrenamiento —prácticamente
memoriza— y aun así queda penúltimo en validación. Con 1166 filas no hay material suficiente
para que el boosting despliegue su ventaja habitual en datos tabulares.

**La regresión sin regularizar no fracasó.** Con 122 columnas y 1166 filas llega a 0.1338, a
0.008 del ganador. El trabajo de los notebooks 1 a 3 dejó un problema donde lo lineal alcanza.

**La curva de aprendizaje del ganador converge** (brecha final 0.0096) pero la de validación
todavía bajaba 0.0021 en el último tramo: más datos ayudarían un poco, aunque el techo está
sobre todo en las features.

**Dos variables cargan con casi todo.** `GrLivArea` y `OverallQual` concentran la mayor parte
de la importancia por permutación; las cinco primeras suman el 70 %. `FullBath`, `FireplaceQu`
y `Exterior2nd` no aportan nada medible.

---

## 8. Problemas encontrados en el camino

| Síntoma | Causa real | Solución |
|---|---|---|
| Kernel muere sin excepción | `Library\bin` fuera del PATH de DLLs | Ejecutar con el entorno activado (`conda run`) |
| Error de CV sospechosamente bajo | Se leía el CSV ya transformado | Partir de `raw` y clonar el pipeline por pliegue |
| Primer punto de la curva de aprendizaje vacío | `CategoricalImputer(frequent)` aborta cuando dos categorías de `Neighborhood` empatan; `learning_curve` lo convierte en `NaN` silencioso | Retirar ese paso del clon de la curva tras probar con `assert` que es no-op (0 faltantes en sus 10 columnas) y usar `error_score="raise"` |
| Óptimo de ElasticNet en el borde de la rejilla | Rejilla demasiado corta | La comprobación automática lo detectó; se amplió de 9 a 21 configuraciones |
| `conda run python -c` falla | No admite argumentos con saltos de línea | Escribir el código a un archivo primero |

---

## 9. Siguiente paso

**Exponer el modelo con FastAPI.** La biblioteca ya está preparada:

- `predecir_registros(registros: list[dict]) -> list[dict]` recibe y devuelve exactamente lo
  que produce el deserializador de JSON, con tipos nativos de Python.
- `cargar_modelo()` está memorizada con `lru_cache`: deserializar tarda ~4 s y predecir 1459
  filas ~0.4 s, así que el servicio debe cargar una sola vez.
- Las excepciones propias (`EntradaInvalida`, `SalidaIncoherente`, `ModeloNoDisponible`)
  mapean naturalmente a códigos HTTP: 422, 500 y 503.
- El parámetro `estricto=True` es el adecuado para un endpoint.

### Deuda técnica pendiente

- `.vscode/settings.json` apunta al entorno viejo.
- No hay kernel de Jupyter registrado para `env_house_pricing`.
- `environment.yml` quedó desincronizado con `requirements.txt`.
- Las funciones del notebook 2 (`imputar`, `codificar_ordinales`, `tratar_garage_yr`) siguen
  en el notebook; deberían migrar a `house_pricing/dataset.py`.
- El proyecto no es un repo de git todavía.
- `data/processed/train_features.csv` sigue existiendo con las transformaciones ajustadas sobre
  las 1458 filas. Quien lo use para entrenar reintroducirá la fuga que el notebook 4.0 evitó.

---

## 10. Convenciones

- **Notebooks:** `<orden>-<iniciales>-<descripción>`, p. ej. `1.0-dg-analisis-exploratorio`.
  Fase 1 = exploración, 2 = limpieza, 3 = features, 4 = modelado, 5 = inferencia.
- **`requirements.txt` es la fuente de verdad.** Toda librería que entre al entorno se agrega
  al archivo en el mismo paso, con versión fijada.
- **Datos:** `raw` inmutable → `interim` (limpio) → `processed` (listo para modelar).
- **Código:** la lógica reutilizable vive en `house_pricing/`; `scripts/` solo tiene puntos de
  entrada. `make lint` cubre el paquete, `scripts/` y `tests/`.
