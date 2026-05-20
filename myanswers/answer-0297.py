import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def calcular_error_precio_casas(X=None, y=None, **kwargs):

    # Por si llegan dentro de kwargs
    if X is None:
        X = kwargs.get("X")

    if y is None:
        y = kwargs.get("y")

    # División entrenamiento/prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    # Modelo
    modelo = LinearRegression()

    # Entrenamiento
    modelo.fit(X_train, y_train)

    # Predicción
    y_pred = modelo.predict(X_test)

    # Error cuadrático medio
    mse = mean_squared_error(y_test, y_pred)

    return float(mse)
