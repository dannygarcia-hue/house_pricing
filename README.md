# house_pricing

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Predicción de precios de vivienda con el dataset *House Prices: Advanced Regression
Techniques* de Kaggle.

**Modelo actual:** ElasticNet (`alpha=0.001`, `l1_ratio=0.5`) · **20 042 USD** de RMSE real,
**4.01 %** del rango, **0.1401** de RMSE logarítmico, medido sobre 292 casas reservadas.

El recorrido completo —qué se hizo, qué se decidió y por qué— está en
[`ESTADO_PROYECTO.md`](ESTADO_PROYECTO.md) y, en formato narrativo, en
[`reports/proceso.docx`](reports/proceso.docx).

## Puesta en marcha

Este repositorio **no versiona datos ni modelos**: `data/` y `models/` están en `.gitignore`
porque son, respectivamente, un dataset público y binarios que se regeneran. Un clon recién
hecho tiene el código pero no puede predecir todavía. Para dejarlo operativo:

```bash
# 1. Entorno
conda create -n env_house_pricing -c conda-forge python=3.11 -y
conda activate env_house_pricing
pip install -r requirements.txt

# 2. Datos: descargar de Kaggle y dejar train.csv y test.csv en data/raw/
#    https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data

# 3. Reconstruir los artefactos, en este orden (unos 10 minutos en total)
jupyter nbconvert --to notebook --execute --inplace notebooks/2.0-dg-limpieza-datos.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/3.0-dg-feature-engineering.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/4.0-dg-modelado.ipynb
```

El paso 3 genera `models/pipeline_features.joblib`, `models/modelo_ganador.joblib` y
`models/modelo_produccion.joblib`. A partir de ahí funcionan tanto las pruebas como la
inferencia.

> **Importante:** ejecutar siempre con el entorno activado. Invocar directamente el
> `python.exe` del entorno mata el proceso sin excepción de Python en cuanto scikit-learn
> toca BLAS. Desde terminal, la forma segura es
> `conda run -n env_house_pricing --no-capture-output python ...`

## Uso

```bash
# Prueba rápida con observaciones al azar del test
python scripts/predecir.py demo --n 3 --semilla 42

# Un CSV completo, en formato Kaggle
python scripts/predecir.py archivo data/raw/test.csv --salida data/processed/submission.csv

# Ficha del modelo cargado
python scripts/predecir.py ficha

# Pruebas y estilo
pytest
make lint
```

Cada ejecución deja su log en `scripts/logs/inferencia_<AAAAMMDD-HHMMSS>.log`.

Para consumir el modelo desde otro proceso —un servicio de FastAPI, por ejemplo— hay que
importar la biblioteca, no el script:

```python
from house_pricing.modeling.predict import predecir_registros

precios = predecir_registros([{"GrLivArea": 1710, "OverallQual": 7, ...}])
```

## Estructura de los notebooks

| Notebook | Fase |
|---|---|
| `1.0-dg-analisis-exploratorio` | Entender los datos |
| `2.0-dg-limpieza-datos` | Imputación y outliers |
| `3.0-dg-feature-engineering` | Pipeline de 13 pasos |
| `4.0-dg-modelado` | Torneo de 8 modelos |
| `5.0-dg-inferencia` | Predicción sobre el test |

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         house_pricing and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── house_pricing   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes house_pricing a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

