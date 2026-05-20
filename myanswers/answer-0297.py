import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# ---------------------------------------------------------
# FUNCIÓN PRINCIPAL
# ---------------------------------------------------------
def calcular_error_precio_casas(X, y):

    # División entrenamiento/prueba (20% prueba)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Modelo de regresión lineal
    modelo = LinearRegression()

    # Entrenamiento
    modelo.fit(X_train, y_train)

    # Predicciones
    y_pred = modelo.predict(X_test)

    # Cálculo del error
    mse = mean_squared_error(y_test, y_pred)

    # Retornar únicamente float
    return float(mse)


# ---------------------------------------------------------
# EJECUCIÓN DEL CASO DE USO
# ---------------------------------------------------------
info, X, y = generar_caso_de_uso_calcular_error_precio_casas()

resultado = calcular_error_precio_casas(X, y)

print(info)
print("MSE:", resultado)
