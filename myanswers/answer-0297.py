import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def calcular_error_precio_casas(*args, **kwargs):
    """
    Toma datos de casas y precios reales, entrena una Regresión Lineal
    y retorna el Error Cuadrático Medio (MSE) en el conjunto de prueba.
    """
    
    # Extraer X e y de args o kwargs
    X = kwargs.get('X', args[0] if len(args) > 0 else None)
    y = kwargs.get('y', args[1] if len(args) > 1 else None)
    
    # Si no se encuentran, buscar en otras claves
    if X is None:
        for key in ['X', 'data', 'features']:
            if key in kwargs:
                X = kwargs[key]
                break
    
    if y is None:
        for key in ['y', 'target', 'precio', 'price']:
            if key in kwargs:
                y = kwargs[key]
                break
    
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
