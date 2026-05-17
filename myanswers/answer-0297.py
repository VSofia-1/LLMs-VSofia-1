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
        test_size=0.20,
        random_state=42
    )

    # Modelo de Regresión Lineal
    modelo = LinearRegression()

    # Entrenamiento
    modelo.fit(X_train, y_train)

    # Predicciones
    y_pred = modelo.predict(X_test)

    # Error Cuadrático Medio
    mse = mean_squared_error(y_test, y_pred)

    return float(mse)
# ---------------------------------------------------------
# EJECUCIÓN
# ---------------------------------------------------------
if __name__ == "__main__":

    # Generar caso de uso
    info, X, y = (
        generar_caso_de_uso_calcular_error_precio_casas()
    )

    # Mostrar información
    print("=== INFORMACIÓN ===")
    print(info)

    print("\n=== DATOS X ===")
    print(X.head())

    print("\n=== DATOS y ===")
    print(y[:5])

    # Llamar implementación
    resultado = calcular_error_precio_casas(X, y)

    # Mostrar resultado
    print("\n=== MSE DEL MODELO ===")
    print(resultado)
