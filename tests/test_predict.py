"""Pruebas de `house_pricing.modeling.predict`.

La muestra de prueba son observaciones reales del test crudo, sorteadas con semilla
fija: así cada ejecución recorre las mismas filas —y los fallos son reproducibles—
pero las filas no se eligieron a mano para que el modelo quedara bien.

Las pruebas se agrupan en cuatro bloques: el contrato de entrada, el camino feliz, la
verificación de calidad y el envoltorio para API.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

# El import va aquí abajo a propósito: necesita el sys.path que se ajustó arriba.
from house_pricing.modeling.predict import (
    EntradaInvalida,
    ModeloNoDisponible,
    SalidaIncoherente,
    cargar_modelo,
    muestrear_test,
    predecir,
    predecir_registros,
    validar_entrada,
    verificar_calidad,
)

SEMILLA = 42
N_MUESTRA = 8


@pytest.fixture(scope="module")
def modelo():
    """El modelo de producción, cargado una sola vez para todo el módulo."""
    return cargar_modelo()


@pytest.fixture(scope="module")
def muestra() -> pd.DataFrame:
    """Ocho observaciones reales del test crudo, siempre las mismas."""
    return muestrear_test(n=N_MUESTRA, semilla=SEMILLA)


# --------------------------------------------------------------------------- #
# Contrato de entrada
# --------------------------------------------------------------------------- #


def test_rechaza_lo_que_no_es_dataframe(modelo):
    """Una lista no es un DataFrame y debe rechazarse antes de tocar el modelo."""
    with pytest.raises(EntradaInvalida, match="DataFrame"):
        validar_entrada([{"GrLivArea": 1710}], modelo.features)


def test_rechaza_entrada_vacia(modelo):
    """Un DataFrame sin filas no produce predicciones: es un error, no un caso borde."""
    with pytest.raises(EntradaInvalida, match="no tiene filas"):
        validar_entrada(pd.DataFrame(columns=modelo.features), modelo.features)


def test_rechaza_columna_faltante(modelo, muestra):
    """Si falta una columna requerida, el error debe nombrarla."""
    incompleta = muestra.drop(columns=["GrLivArea"])
    with pytest.raises(EntradaInvalida, match="GrLivArea"):
        validar_entrada(incompleta, modelo.features)


def test_ignora_columnas_de_mas(modelo, muestra):
    """El test crudo trae 80 columnas y el modelo usa 30: las otras 50 se descartan."""
    validada = validar_entrada(muestra, modelo.features)
    assert list(validada.columns) == modelo.features


def test_reordena_columnas_desordenadas(modelo, muestra):
    """Unas columnas en otro orden deben volver al orden del entrenamiento."""
    desordenada = muestra.loc[:, modelo.features[::-1]]
    validada = validar_entrada(desordenada, modelo.features)
    assert list(validada.columns) == modelo.features


def test_el_orden_de_columnas_no_cambia_la_prediccion(modelo, muestra):
    """La misma fila, con las columnas al revés, debe predecir exactamente lo mismo."""
    normal = predecir(muestra, modelo=modelo)
    invertida = predecir(muestra.loc[:, modelo.features[::-1]], modelo=modelo)
    assert np.allclose(normal["SalePrice"], invertida["SalePrice"])


def test_artefacto_inexistente():
    """Pedir un modelo que no existe debe fallar con un mensaje accionable."""
    with pytest.raises(ModeloNoDisponible, match="no existe"):
        cargar_modelo(RAIZ / "models" / "no_existe.joblib")


# --------------------------------------------------------------------------- #
# Camino feliz
# --------------------------------------------------------------------------- #


def test_una_fila_suelta(modelo, muestra):
    """Predecir sobre una sola observación, que es el caso de uso de la API."""
    resultado = predecir(muestra.head(1), modelo=modelo)
    assert len(resultado) == 1
    assert resultado["coherente"].all()
    assert resultado["SalePrice"].iloc[0] > 0


def test_lote_completo(modelo, muestra):
    """Sale exactamente una predicción por fila, con el índice de la entrada."""
    resultado = predecir(muestra, modelo=modelo)
    assert len(resultado) == len(muestra)
    assert list(resultado.index) == list(muestra.index)
    assert list(resultado.columns) == ["SalePrice", "prediccion_log", "coherente", "motivo"]


def test_predicciones_en_rango_plausible(modelo, muestra):
    """Ninguna casa del test debería predecirse fuera del orden de magnitud conocido."""
    resultado = predecir(muestra, modelo=modelo)
    assert (resultado["SalePrice"] > 10_000).all()
    assert (resultado["SalePrice"] < 1_500_000).all()


def test_expm1_deshace_el_logaritmo(modelo, muestra):
    """La columna en dólares debe ser exactamente expm1 de la columna en logaritmo."""
    resultado = predecir(muestra, modelo=modelo)
    assert np.allclose(resultado["SalePrice"], np.expm1(resultado["prediccion_log"]))


def test_es_determinista(modelo, muestra):
    """Dos llamadas con los mismos datos dan el mismo resultado."""
    primera = predecir(muestra, modelo=modelo)
    segunda = predecir(muestra, modelo=modelo)
    assert np.allclose(primera["SalePrice"], segunda["SalePrice"])


def test_sobrevive_a_los_faltantes(modelo, muestra):
    """Vaciar columnas enteras no debe romper nada: el pipeline las imputa."""
    con_huecos = muestra.copy()
    con_huecos.loc[:, ["LotFrontage", "GarageYrBlt", "MasVnrArea", "KitchenQual"]] = np.nan

    resultado = predecir(con_huecos, modelo=modelo)
    assert len(resultado) == len(muestra)
    assert resultado["coherente"].all()


def test_sobrevive_a_una_categoria_desconocida(modelo, muestra):
    """Un barrio que el modelo nunca vio no debe tumbar la predicción."""
    inventada = muestra.copy()
    inventada["Neighborhood"] = "BarrioQueNoExiste"

    resultado = predecir(inventada, modelo=modelo)
    assert len(resultado) == len(muestra)
    assert resultado["coherente"].all()


def test_sobrevive_al_anio_de_garaje_imposible(modelo, muestra):
    """El test real trae un GarageYrBlt de 2207; el pipeline lo corrige solo."""
    rara = muestra.copy()
    rara["GarageYrBlt"] = 2207.0

    resultado = predecir(rara, modelo=modelo)
    assert resultado["coherente"].all()


# --------------------------------------------------------------------------- #
# Verificación de calidad
# --------------------------------------------------------------------------- #


def test_calidad_cuenta_las_filas_sanas(muestra):
    """Con predicciones normales, todas las filas se cuentan como coherentes."""
    precios = np.full(len(muestra), 180_000.0)
    resultado, coherentes, motivos = verificar_calidad(muestra, precios)

    assert resultado.todo_coherente
    assert resultado.filas_coherentes == len(muestra)
    assert resultado.filas_incoherentes == 0
    assert coherentes.all()
    assert motivos == [""] * len(muestra)


@pytest.mark.parametrize(
    ("valor", "motivo_esperado"),
    [
        (np.nan, "nula"),
        (-5_000.0, "no_positiva"),
        (50.0, "fuera_de_rango"),
        (9_000_000.0, "fuera_de_rango"),
    ],
)
def test_calidad_detecta_cada_tipo_de_problema(muestra, valor, motivo_esperado):
    """Cada clase de predicción imposible debe quedar marcada con su motivo."""
    precios = np.full(len(muestra), 180_000.0)
    precios[0] = valor

    resultado, coherentes, motivos = verificar_calidad(muestra, precios)

    assert not resultado.todo_coherente
    assert resultado.filas_incoherentes == 1
    assert not coherentes[0]
    assert motivo_esperado in motivos[0]
    assert coherentes[1:].all()


def test_calidad_falla_si_se_pierden_filas(muestra):
    """Que entren 8 filas y salgan 7 es un fallo estructural, no una fila marcada."""
    with pytest.raises(SalidaIncoherente, match="perdió o duplicó"):
        verificar_calidad(muestra, np.full(len(muestra) - 1, 180_000.0))


def test_estricto_true_levanta_excepcion(modelo, muestra, monkeypatch):
    """Con estricto=True, una predicción imposible debe abortar la llamada."""

    def predecir_basura(_entrada):
        return np.full(len(muestra), -3.0)

    monkeypatch.setattr(modelo.pipeline, "predict", predecir_basura)

    with pytest.raises(SalidaIncoherente, match="no utilizable|no\\s+son utilizables"):
        predecir(muestra, modelo=modelo, estricto=True)


def test_estricto_false_devuelve_las_filas_marcadas(modelo, muestra, monkeypatch):
    """Con estricto=False, las mismas predicciones vuelven marcadas en vez de estallar."""

    def predecir_basura(_entrada):
        return np.full(len(muestra), -3.0)

    monkeypatch.setattr(modelo.pipeline, "predict", predecir_basura)

    resultado = predecir(muestra, modelo=modelo, estricto=False)
    assert len(resultado) == len(muestra)
    assert not resultado["coherente"].any()
    assert (resultado["motivo"] != "").all()


# --------------------------------------------------------------------------- #
# Envoltorio para API
# --------------------------------------------------------------------------- #


def test_registros_devuelve_json_serializable(modelo, muestra):
    """FastAPI necesita tipos nativos: nada de float64 ni bool_ de NumPy."""
    registros = muestra.loc[:, modelo.features].to_dict(orient="records")
    salida = predecir_registros(registros, modelo=modelo)

    assert len(salida) == len(registros)
    for fila in salida:
        assert type(fila["SalePrice"]) is float
        assert type(fila["prediccion_log"]) is float
        assert type(fila["coherente"]) is bool
        assert type(fila["motivo"]) is str


def test_registros_coincide_con_dataframe(modelo, muestra):
    """Las dos puertas de entrada deben dar exactamente el mismo precio."""
    registros = muestra.loc[:, modelo.features].to_dict(orient="records")

    por_registros = [fila["SalePrice"] for fila in predecir_registros(registros, modelo=modelo)]
    por_dataframe = predecir(muestra, modelo=modelo)["SalePrice"].tolist()

    assert np.allclose(por_registros, por_dataframe)


def test_registros_rechaza_lista_vacia(modelo):
    """Una petición sin registros es un error del cliente, no una respuesta vacía."""
    with pytest.raises(EntradaInvalida, match="al menos un registro"):
        predecir_registros([], modelo=modelo)


def test_registros_rechaza_elementos_que_no_son_dict(modelo):
    """Una lista con elementos sueltos debe señalar cuáles están mal."""
    with pytest.raises(EntradaInvalida, match="no son diccionarios"):
        predecir_registros([{"GrLivArea": 1710}, 42], modelo=modelo)


# --------------------------------------------------------------------------- #
# Muestreo
# --------------------------------------------------------------------------- #


def test_muestreo_con_semilla_es_reproducible():
    """La misma semilla debe devolver las mismas casas."""
    primera = muestrear_test(n=5, semilla=SEMILLA)
    segunda = muestrear_test(n=5, semilla=SEMILLA)
    assert primera["Id"].tolist() == segunda["Id"].tolist()


def test_muestreo_pide_mas_filas_de_las_que_hay():
    """Pedir más observaciones que las del test es un error explícito."""
    with pytest.raises(EntradaInvalida, match="solo tiene"):
        muestrear_test(n=10_000)
