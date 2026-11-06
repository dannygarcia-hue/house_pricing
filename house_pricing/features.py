from pathlib import Path

from loguru import logger
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm
import typer

from house_pricing.config import PROCESSED_DATA_DIR

app = typer.Typer()


class RellenarAnioGaraje(BaseEstimator, TransformerMixin):
    """Rellena `GarageYrBlt` con `YearBuilt` y corrige años posteriores a la venta.

    Un `GarageYrBlt` nulo significa que la casa no tiene garaje: no es un dato
    perdido ni un cero. Imputar 0 inventaría casas construidas en el año cero y
    arruinaría cualquier modelo lineal. `YearBuilt` correlaciona 0.83 con esta
    columna, así que sirve de relleno; el indicador que añade `AddMissingIndicator`
    antes de este paso deja constancia de qué filas no tenían garaje.

    Vive en el paquete y no en el notebook para que el pipeline serializado con
    joblib se pueda recargar desde otro proceso.
    """

    def __init__(self, anio_maximo: int = 2010):
        self.anio_maximo = anio_maximo

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["GarageYrBlt"] = X["GarageYrBlt"].fillna(X["YearBuilt"])
        # En test hay un registro con GarageYrBlt = 2207: error de captura.
        fuera_de_rango = X["GarageYrBlt"] > self.anio_maximo
        X.loc[fuera_de_rango, "GarageYrBlt"] = X.loc[fuera_de_rango, "YearBuilt"]
        return X


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "features.csv",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Generating features from dataset...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Features generation complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
