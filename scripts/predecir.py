"""Punto de entrada por línea de comandos para la inferencia de House Prices.

Este archivo no implementa nada: traduce argumentos de consola a llamadas de
`house_pricing.modeling.predict`, que es donde vive la lógica. Esa separación existe
porque el servicio de FastAPI que se construya después debe importar las mismas
funciones que usa este script, y no una copia que se desincronice con el tiempo.

Modos de uso::

    # Prueba rápida con 5 observaciones al azar del test
    python scripts/predecir.py demo

    # Siempre las mismas 3 filas, para comparar entre ejecuciones
    python scripts/predecir.py demo --n 3 --semilla 42

    # Un CSV completo, en formato Kaggle
    python scripts/predecir.py archivo data/raw/test.csv --salida data/processed/submission.csv

Cada ejecución deja su propio log en `scripts/logs/inferencia_<AAAAMMDD-HHMMSS>.log`.
"""

from pathlib import Path
import sys

from loguru import logger
import pandas as pd
import typer

# El paquete vive un nivel arriba de scripts/. Esta línea hace que el script funcione
# aunque house_pricing no esté instalado en el entorno.
RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

# El import va aquí abajo a propósito: necesita el sys.path que se ajustó arriba.
from house_pricing.modeling.predict import (
    ErrorDeInferencia,
    cargar_modelo,
    configurar_logging,
    muestrear_test,
    predecir,
    predecir_csv,
)

CARPETA_LOGS: Path = Path(__file__).resolve().parent / "logs"

app = typer.Typer(add_completion=False, help="Inferencia con el modelo ganador de House Prices.")


def _resumir(entrada: pd.DataFrame, resultado: pd.DataFrame, columnas: list[str]) -> None:
    """Imprime entrada y predicción lado a lado, para inspección a ojo.

    Args:
        entrada: Las filas que se enviaron al modelo.
        resultado: Lo que devolvió `predecir`, con el mismo índice.
        columnas: Qué columnas de la entrada mostrar junto al precio.
    """
    visibles = [columna for columna in columnas if columna in entrada.columns]
    tabla = entrada.loc[:, visibles].copy()
    tabla["PRECIO_PREDICHO"] = resultado["SalePrice"].round(0).astype(int)
    tabla["coherente"] = resultado["coherente"]

    typer.echo("\n" + tabla.to_string())
    typer.echo(
        f"\nmediana {resultado['SalePrice'].median():,.0f} USD · "
        f"rango {resultado['SalePrice'].min():,.0f} a {resultado['SalePrice'].max():,.0f} USD"
    )


@app.command()
def demo(
    n: int = typer.Option(5, help="Cuántas observaciones del test sortear."),
    semilla: int = typer.Option(
        None, help="Semilla del muestreo. Sin ella, filas distintas cada vez."
    ),
    estricto: bool = typer.Option(
        True, help="Abortar si alguna predicción no supera las comprobaciones de calidad."
    ),
) -> None:
    """Predice sobre observaciones al azar del test crudo.

    Es el mecanismo de prueba del script: no necesita preparar ningún archivo, usa
    `data/raw/test.csv` tal cual y enseña por consola qué entró y qué salió.
    """
    configurar_logging(CARPETA_LOGS)
    logger.info(f"modo demo · {n} observación(es) al azar · semilla {semilla}")

    try:
        modelo = cargar_modelo()
        muestra = muestrear_test(n=n, semilla=semilla)
        resultado = predecir(muestra, modelo=modelo, estricto=estricto)
    except ErrorDeInferencia as error:
        logger.error(f"la demo falló: {error}")
        raise typer.Exit(code=1) from error

    _resumir(
        muestra,
        resultado,
        ["Id", "Neighborhood", "OverallQual", "GrLivArea", "YearBuilt", "GarageCars"],
    )
    logger.success("demo terminada")


@app.command()
def archivo(
    entrada: Path = typer.Argument(..., help="CSV con las columnas crudas."),
    salida: Path = typer.Option(
        None, help="Dónde escribir las predicciones. Sin esto, solo se muestran."
    ),
    estricto: bool = typer.Option(
        True, help="Abortar si alguna predicción no supera las comprobaciones de calidad."
    ),
) -> None:
    """Predice sobre un CSV completo y escribe el resultado en formato Kaggle."""
    configurar_logging(CARPETA_LOGS)
    logger.info(f"modo archivo · entrada {entrada} · salida {salida}")

    try:
        resultado = predecir_csv(entrada, salida, estricto=estricto)
    except ErrorDeInferencia as error:
        logger.error(f"la inferencia falló: {error}")
        raise typer.Exit(code=1) from error

    typer.echo("\n" + resultado.head(10).to_string())
    incoherentes = int((~resultado["coherente"]).sum())
    typer.echo(
        f"\n{len(resultado):,} predicción(es) · {incoherentes} incoherente(s) · "
        f"mediana {resultado['SalePrice'].median():,.0f} USD"
    )
    logger.success("inferencia terminada")


@app.command()
def ficha() -> None:
    """Muestra la ficha del artefacto cargado, sin predecir nada."""
    configurar_logging(CARPETA_LOGS)

    try:
        modelo = cargar_modelo()
    except ErrorDeInferencia as error:
        logger.error(str(error))
        raise typer.Exit(code=1) from error

    typer.echo(f"\nmodelo        {modelo.nombre}")
    typer.echo(f"configuración {modelo.configuracion}")
    typer.echo(f"ajustado con  {modelo.ajustado_con}")
    typer.echo(f"artefacto     {modelo.ruta}")
    typer.echo(f"features      {len(modelo.features)}")
    for clave, valor in modelo.error_esperado.items():
        typer.echo(f"  {clave:12} {valor:,.4f}" if valor < 1 else f"  {clave:12} {valor:,.0f}")


if __name__ == "__main__":
    app()
