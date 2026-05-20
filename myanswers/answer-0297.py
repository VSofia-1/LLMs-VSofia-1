import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def calcular_error_precio_casas(X, y, **kwargs):
    """
    Toma datos de casas y precios reales, entrena una Regresión Lineal
    y retorna el Error Cuadrático Medio (MSE) en el conjunto de prueba.

    Args:
        X (pd.DataFrame o np.ndarray): Variables predictoras.
        y (pd.Series o np.ndarray): Precio real de las casas.
        **kwargs: Argumentos adicionales (ignorados para compatibilidad).

    Returns:
        float: Error Cuadrático Medio (MSE) del modelo.
    """
    
    # 1. Dividir los datos: 80% entrenamiento, 20% prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 2. Entrenar el modelo de Regresión Lineal
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)

    # 3. Realizar predicciones sobre el conjunto de prueba
    predicciones = modelo.predict(X_test)

    # 4. Calcular el Error Cuadrático Medio (MSE)
    error_mse = mean_squared_error(y_test, predicciones)

    return float(error_mse)
