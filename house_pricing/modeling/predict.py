"""Inferencia con el modelo ganador de House Prices.

Este módulo es la biblioteca: expone las funciones que ejecutan una predicción y
verifican su calidad. El punto de entrada por línea de comandos vive en
`scripts/predecir.py`, y el servicio de FastAPI que se construya más adelante debe
importar de aquí, no duplicar la lógica.

El artefacto que se carga es `models/modelo_produccion.joblib`, un `Pipeline` de
scikit-learn con dos pasos: las 13 transformaciones de ingeniería de características
del notebook 3.0 y el ElasticNet que ganó el torneo del notebook 4.0. Eso fija el
contrato:

- **Entra** un `DataFrame` con las 30 columnas crudas, tal como vienen de
  `data/raw/test.csv`: con faltantes, con texto y sin escalar. El pipeline imputa,
  codifica, recorta, transforma y escala por su cuenta.
- **Sale** `log1p(SalePrice)`, que este módulo convierte a dólares con `numpy.expm1`.

Ejemplo de uso desde otro proceso::

    from house_pricing.modeling.predict import predecir_registros

    precios = predecir_registros([{"GrLivArea": 1710, "OverallQual": 7, ...}])
"""

from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from loguru import logger
import numpy as np
import pandas as pd

from house_pricing.config import MODELS_DIR

RUTA_MODELO_POR_DEFECTO: Path = MODELS_DIR / "modelo_produccion.joblib"

# Límites de lo que se considera un precio posible. No salen de una regla de negocio
# sino de los datos: en train el rango real es 34 900 a 755 000 dólares, y estos
# límites lo ensanchan con holgura. Una predicción fuera de aquí no es "cara" o
# "barata": es la señal de que algo se rompió antes, normalmente una columna mal
# alineada o un valor que el pipeline no supo tratar.
PRECIO_MINIMO_PLAUSIBLE: float = 10_000.0
PRECIO_MAXIMO_PLAUSIBLE: float = 1_500_000.0

# Formato de la marca de tiempo de los archivos de log.
FORMATO_ESTAMPA: str = "%Y%m%d-%H%M%S"


class ErrorDeInferencia(Exception):
    """Error base del módulo. Permite capturar cualquier fallo propio con un solo `except`."""


class ModeloNoDisponible(ErrorDeInferencia):
    """El artefacto del modelo no existe o no se pudo cargar."""


class EntradaInvalida(ErrorDeInferencia):
    """Los datos recibidos no cumplen el contrato de entrada del modelo."""


class SalidaIncoherente(ErrorDeInferencia):
    """El modelo respondió, pero su salida no supera las comprobaciones de calidad."""


@dataclass(frozen=True)
class ModeloCargado:
    """El pipeline listo para predecir, junto con la ficha de su artefacto.

    Attributes:
        pipeline: El `Pipeline` de scikit-learn ajustado, de columnas crudas a `log1p`.
        nombre: Familia del modelo final, por ejemplo ``"ElasticNet"``.
        configuracion: Hiperparámetros elegidos en el torneo, como texto legible.
        features: Las 30 columnas de entrada, en el orden en que se entrenó.
        ajustado_con: Descripción de los datos con los que se ajustó.
        error_esperado: RMSE logarítmico, en dólares y normalizado al rango.
        ruta: De dónde se cargó el artefacto.
    """

    pipeline: Any
    nombre: str
    configuracion: str
    features: list[str]
    ajustado_con: str
    error_esperado: dict[str, float]
    ruta: Path


@dataclass(frozen=True)
class ResultadoCalidad:
    """Recuento de la comprobación entre lo que entró y lo que salió.

    Attributes:
        filas_entrada: Cuántas filas se recibieron.
        filas_salida: Cuántas predicciones se produjeron.
        filas_coherentes: Cuántas predicciones superan todas las comprobaciones.
        motivos: Cuántas filas falló cada comprobación, por nombre de la comprobación.
    """

    filas_entrada: int
    filas_salida: int
    filas_coherentes: int
    motivos: dict[str, int]

    @property
    def filas_incoherentes(self) -> int:
        """Número de predicciones que fallaron al menos una comprobación."""
        return self.filas_salida - self.filas_coherentes

    @property
    def todo_coherente(self) -> bool:
        """`True` si cada fila de entrada produjo una predicción utilizable."""
        return (
            self.filas_entrada == self.filas_salida
            and self.filas_coherentes == self.filas_salida
            and self.filas_entrada > 0
        )


def configurar_logging(carpeta: Path, nivel: str = "INFO") -> Path:
    """Añade un archivo de log con marca de tiempo, sin tocar la salida por consola.

    Se usa `logger.add`, no `logger.configure`, para que el sink que `house_pricing.config`
    instala al importarse siga funcionando: la consola conserva su formato y el archivo
    se suma como destino adicional.

    Args:
        carpeta: Dónde escribir el archivo. Se crea si no existe.
        nivel: Nivel mínimo que se registra en el archivo.

    Returns:
        La ruta del archivo creado, con la forma ``inferencia_<AAAAMMDD-HHMMSS>.log``.
    """
    carpeta.mkdir(parents=True, exist_ok=True)
    archivo = carpeta / f"inferencia_{datetime.now().strftime(FORMATO_ESTAMPA)}.log"
    logger.add(
        archivo,
        level=nivel,
        encoding="utf-8",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
    )
    logger.info(f"log de esta ejecución: {archivo}")
    return archivo


@lru_cache(maxsize=4)
def cargar_modelo(ruta: Path = RUTA_MODELO_POR_DEFECTO) -> ModeloCargado:
    """Carga el artefacto del modelo desde disco y lo deja en caché.

    El resultado se memoriza con `lru_cache` porque deserializar el pipeline cuesta
    bastante más que predecir con él: un servicio que atiende peticiones debe pagar
    ese costo una vez, no en cada llamada. Para forzar una recarga —tras reentrenar,
    por ejemplo— hay que llamar a ``cargar_modelo.cache_clear()``.

    Args:
        ruta: Ubicación del `.joblib`. Por defecto `models/modelo_produccion.joblib`.

    Returns:
        El modelo y su ficha.

    Raises:
        ModeloNoDisponible: Si el archivo no existe, no se puede deserializar o no
            tiene la estructura esperada.
    """
    if not ruta.exists():
        raise ModeloNoDisponible(
            f"no existe el artefacto {ruta}. Hay que ejecutar la sección 10.1 del "
            "notebook 4.0-dg-modelado para generarlo."
        )

    try:
        import joblib

        artefacto = joblib.load(ruta)
    except Exception as error:  # se captura amplio y se re-lanza con contexto propio
        raise ModeloNoDisponible(f"no se pudo deserializar {ruta}: {error}") from error

    faltan = {"pipeline", "features", "nombre"} - set(artefacto)
    if faltan:
        raise ModeloNoDisponible(f"al artefacto {ruta} le faltan las claves {sorted(faltan)}")

    modelo = ModeloCargado(
        pipeline=artefacto["pipeline"],
        nombre=artefacto["nombre"],
        configuracion=artefacto.get("mejor_config", "desconocida"),
        features=list(artefacto["features"]),
        ajustado_con=artefacto.get("ajustado_con", "desconocido"),
        error_esperado=artefacto.get("error_esperado", {}),
        ruta=ruta,
    )
    logger.info(
        f"modelo cargado: {modelo.nombre} ({modelo.configuracion}) "
        f"· {len(modelo.features)} features · ajustado con {modelo.ajustado_con}"
    )
    return modelo


def validar_entrada(datos: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Comprueba el contrato de entrada y devuelve las columnas en el orden correcto.

    Reordenar no es cosmético: varios transformadores de feature-engine y el escalado
    final operan por posición una vez dentro del pipeline, de modo que unas columnas
    desordenadas producirían predicciones plausibles y equivocadas, que es el peor
    tipo de error porque no se nota.

    Args:
        datos: Observaciones a predecir, una por fila.
        features: Las columnas que el modelo espera, en el orden del entrenamiento.

    Returns:
        Una copia de `datos` limitada a `features` y en ese orden.

    Raises:
        EntradaInvalida: Si no es un `DataFrame`, si viene vacío o si falta alguna
            columna requerida.
    """
    if not isinstance(datos, pd.DataFrame):
        raise EntradaInvalida(f"se esperaba un DataFrame, llegó {type(datos).__name__}")

    if datos.empty:
        raise EntradaInvalida("el conjunto de entrada no tiene filas")

    ausentes = [columna for columna in features if columna not in datos.columns]
    if ausentes:
        raise EntradaInvalida(
            f"faltan {len(ausentes)} columnas requeridas: {ausentes}. "
            f"El modelo necesita exactamente estas {len(features)}: {features}"
        )

    sobrantes = [columna for columna in datos.columns if columna not in features]
    if sobrantes:
        # Solo el recuento y una muestra: con el test crudo sobran 50 columnas y
        # listarlas enteras cada vez vuelve el log ilegible.
        logger.debug(
            f"se ignoran {len(sobrantes)} columnas que el modelo no usa (p. ej. {sobrantes[:5]})"
        )

    return datos.loc[:, features].copy()


def verificar_calidad(
    entrada: pd.DataFrame,
    predicciones: np.ndarray,
    *,
    minimo: float = PRECIO_MINIMO_PLAUSIBLE,
    maximo: float = PRECIO_MAXIMO_PLAUSIBLE,
) -> tuple[ResultadoCalidad, np.ndarray, list[str]]:
    """Contrasta la salida contra la entrada y marca cada fila como coherente o no.

    Son cuatro comprobaciones por fila —sin nulos, finita, positiva y dentro del rango
    plausible— más una estructural: que salga exactamente una predicción por cada fila
    que entró. La estructural es la única que no admite matices y por eso se lanza como
    excepción aquí mismo; las otras cuatro marcan la fila y dejan que quien llama decida.

    Args:
        entrada: Las filas que se enviaron al modelo.
        predicciones: Los precios en dólares que devolvió, uno por fila.
        minimo: Precio mínimo que se considera posible.
        maximo: Precio máximo que se considera posible.

    Returns:
        Una tupla con el recuento, el vector booleano de filas coherentes y la lista
        de motivos por fila (cadena vacía cuando la fila está bien).

    Raises:
        SalidaIncoherente: Si el número de predicciones no coincide con el de filas.
    """
    if len(predicciones) != len(entrada):
        raise SalidaIncoherente(
            f"entraron {len(entrada)} filas y salieron {len(predicciones)} predicciones: "
            "el pipeline perdió o duplicó observaciones"
        )

    comprobaciones: dict[str, np.ndarray] = {
        "nula": pd.isna(predicciones),
        "no_finita": ~np.isfinite(np.nan_to_num(predicciones, nan=0.0)),
        "no_positiva": np.nan_to_num(predicciones, nan=0.0) <= 0,
        "fuera_de_rango": (np.nan_to_num(predicciones, nan=0.0) < minimo)
        | (np.nan_to_num(predicciones, nan=0.0) > maximo),
    }

    motivos: list[str] = []
    for i in range(len(predicciones)):
        fallos = [nombre for nombre, mascara in comprobaciones.items() if bool(mascara[i])]
        motivos.append(", ".join(fallos))

    coherentes = np.array([not motivo for motivo in motivos], dtype=bool)

    resultado = ResultadoCalidad(
        filas_entrada=len(entrada),
        filas_salida=len(predicciones),
        filas_coherentes=int(coherentes.sum()),
        motivos={
            nombre: int(mascara.sum())
            for nombre, mascara in comprobaciones.items()
            if int(mascara.sum()) > 0
        },
    )

    if resultado.todo_coherente:
        logger.info(
            f"calidad: {resultado.filas_coherentes}/{resultado.filas_entrada} filas "
            "con resultado coherente"
        )
    else:
        logger.error(
            f"calidad: {resultado.filas_coherentes}/{resultado.filas_entrada} filas "
            f"coherentes · {resultado.filas_incoherentes} con problemas · "
            f"motivos: {resultado.motivos}"
        )

    return resultado, coherentes, motivos


def predecir(
    datos: pd.DataFrame,
    *,
    modelo: ModeloCargado | None = None,
    estricto: bool = True,
) -> pd.DataFrame:
    """Predice el precio de venta de cada fila y verifica la calidad del resultado.

    Args:
        datos: Observaciones con las 30 columnas crudas que el modelo espera. Puede
            traer columnas de más, que se ignoran, y faltantes, que el pipeline imputa.
        modelo: Modelo ya cargado. Si se omite se usa el de producción, en caché.
        estricto: Si es `True`, cualquier fila incoherente aborta la llamada. Si es
            `False`, se devuelven todas las filas marcadas y quien llama decide. Un
            endpoint de FastAPI probablemente quiera `True`; un proceso por lotes que
            no deba caerse por una fila, `False`.

    Returns:
        Un `DataFrame` con el mismo índice que `datos` y cuatro columnas:
        `SalePrice` (dólares), `prediccion_log` (la salida cruda del modelo),
        `coherente` (bool) y `motivo` (vacío si la fila está bien).

    Raises:
        EntradaInvalida: Si los datos no cumplen el contrato de entrada.
        SalidaIncoherente: Si el pipeline pierde filas, o si `estricto` es `True` y
            alguna predicción no supera las comprobaciones de calidad.
        ModeloNoDisponible: Si hay que cargar el modelo y el artefacto no está.
    """
    modelo = modelo or cargar_modelo()
    entrada = validar_entrada(datos, modelo.features)

    nulos = int(entrada.isna().sum().sum())
    logger.info(
        f"prediciendo {len(entrada)} fila(s) con {modelo.nombre} "
        f"· {nulos} valor(es) faltante(s) que el pipeline imputará"
    )

    try:
        prediccion_log = np.asarray(modelo.pipeline.predict(entrada), dtype=float)
    except Exception as error:  # se captura amplio y se re-lanza con contexto propio
        logger.error(f"el pipeline falló al predecir: {error}")
        raise SalidaIncoherente(f"el pipeline falló al predecir: {error}") from error

    # El modelo se entrenó sobre log1p(SalePrice); expm1 es su inversa exacta.
    precios = np.expm1(prediccion_log)

    calidad, coherentes, motivos = verificar_calidad(entrada, precios)

    if estricto and not calidad.todo_coherente:
        filas_malas = entrada.index[~coherentes].tolist()
        raise SalidaIncoherente(
            f"{calidad.filas_incoherentes} de {calidad.filas_entrada} predicciones no "
            f"son utilizables (motivos: {calidad.motivos}; filas: {filas_malas[:10]}). "
            "Usar estricto=False para recibirlas marcadas en vez de esta excepción."
        )

    resultado = pd.DataFrame(
        {
            "SalePrice": precios,
            "prediccion_log": prediccion_log,
            "coherente": coherentes,
            "motivo": motivos,
        },
        index=entrada.index,
    )

    logger.success(
        f"{len(resultado)} predicción(es) lista(s) · "
        f"mediana {np.median(precios):,.0f} USD · "
        f"rango {precios.min():,.0f} a {precios.max():,.0f} USD"
    )
    return resultado


def predecir_registros(
    registros: list[dict[str, Any]],
    *,
    modelo: ModeloCargado | None = None,
    estricto: bool = True,
) -> list[dict[str, Any]]:
    """Versión de `predecir` con la forma que usa una API JSON.

    Es el envoltorio pensado para FastAPI: recibe lo que produce el deserializador de
    JSON —una lista de diccionarios— y devuelve algo que el serializador puede
    escribir de vuelta, con tipos nativos de Python en lugar de escalares de NumPy.

    Args:
        registros: Una observación por diccionario, con las 30 columnas como claves.
        modelo: Modelo ya cargado. Si se omite se usa el de producción, en caché.
        estricto: Igual que en `predecir`.

    Returns:
        Una lista con un diccionario por registro: ``{"SalePrice": float,
        "prediccion_log": float, "coherente": bool, "motivo": str}``.

    Raises:
        EntradaInvalida: Si la lista viene vacía o sus elementos no son diccionarios.
        SalidaIncoherente: Igual que en `predecir`.
        ModeloNoDisponible: Igual que en `predecir`.
    """
    if not isinstance(registros, list) or not registros:
        raise EntradaInvalida("se esperaba una lista de diccionarios con al menos un registro")

    no_dicts = [i for i, registro in enumerate(registros) if not isinstance(registro, dict)]
    if no_dicts:
        raise EntradaInvalida(f"los elementos {no_dicts[:5]} de la lista no son diccionarios")

    resultado = predecir(pd.DataFrame(registros), modelo=modelo, estricto=estricto)

    return [
        {
            "SalePrice": float(fila.SalePrice),
            "prediccion_log": float(fila.prediccion_log),
            "coherente": bool(fila.coherente),
            "motivo": str(fila.motivo),
        }
        for fila in resultado.itertuples()
    ]


def predecir_csv(
    ruta_entrada: Path,
    ruta_salida: Path | None = None,
    *,
    modelo: ModeloCargado | None = None,
    estricto: bool = True,
) -> pd.DataFrame:
    """Predice sobre un CSV y, si se indica, escribe el resultado en formato Kaggle.

    Args:
        ruta_entrada: CSV con las columnas crudas. Si trae una columna `Id`, se
            conserva para identificar las filas de la salida.
        ruta_salida: Dónde escribir el CSV de predicciones. Si es `None` no se escribe.
        modelo: Modelo ya cargado. Si se omite se usa el de producción, en caché.
        estricto: Igual que en `predecir`.

    Returns:
        Un `DataFrame` con la columna `Id` —cuando existe en la entrada— y las cuatro
        columnas que devuelve `predecir`.

    Raises:
        EntradaInvalida: Si el archivo no existe o su contenido no cumple el contrato.
        SalidaIncoherente: Igual que en `predecir`.
        ModeloNoDisponible: Igual que en `predecir`.
    """
    if not ruta_entrada.exists():
        raise EntradaInvalida(f"no existe el archivo de entrada {ruta_entrada}")

    datos = pd.read_csv(ruta_entrada)
    logger.info(f"leído {ruta_entrada} · {len(datos)} filas × {datos.shape[1]} columnas")

    resultado = predecir(datos, modelo=modelo, estricto=estricto)

    if "Id" in datos.columns:
        resultado.insert(0, "Id", datos["Id"].to_numpy())

    if ruta_salida is not None:
        ruta_salida.parent.mkdir(parents=True, exist_ok=True)
        columnas = ["Id", "SalePrice"] if "Id" in resultado.columns else ["SalePrice"]
        resultado.loc[:, columnas].to_csv(ruta_salida, index=False)
        logger.success(f"escrito {ruta_salida} · {len(resultado)} filas")

    return resultado


def muestrear_test(
    n: int = 5,
    *,
    semilla: int | None = None,
    ruta_test: Path | None = None,
) -> pd.DataFrame:
    """Toma observaciones al azar del test crudo, para probar el script.

    Args:
        n: Cuántas filas devolver.
        semilla: Semilla del muestreo. `None` da una muestra distinta cada vez.
        ruta_test: CSV del que muestrear. Por defecto `data/raw/test.csv`.

    Returns:
        Las `n` filas sorteadas, con todas sus columnas originales incluido `Id`.

    Raises:
        EntradaInvalida: Si el archivo no existe o se piden más filas de las que tiene.
    """
    from house_pricing.config import RAW_DATA_DIR

    ruta_test = ruta_test or RAW_DATA_DIR / "test.csv"
    if not ruta_test.exists():
        raise EntradaInvalida(f"no existe el archivo de test {ruta_test}")

    test = pd.read_csv(ruta_test)
    if n > len(test):
        raise EntradaInvalida(f"se pidieron {n} filas y el test solo tiene {len(test)}")

    muestra = test.sample(n=n, random_state=semilla)
    logger.info(
        f"muestra de {n} fila(s) de {ruta_test.name} "
        f"(semilla {semilla}) · Id: {muestra['Id'].tolist()}"
    )
    return muestra
