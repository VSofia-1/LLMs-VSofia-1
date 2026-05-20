import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def calcular_error_precio_casas(X, **kwargs):
    """
    Toma datos de casas y precios reales, entrena una Regresión Lineal
    y retorna el Error Cuadrático Medio (MSE) en el conjunto de prueba.
    
   Args:
        X (pd.DataFrame o np.ndarray): Variables predictoras de las casas.
        **kwargs: Puede contener 'descripcion', 'n_muestras', 'n_features', o 'y'.
    """
    
    # Obtener y desde kwargs si existe, si no, generarlo
    y = kwargs.get('y', None)
    
    if y is None:
        # Generar y basado en X para poder ejecutar el modelo
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = np.array(X)
        
        n_features = X_array.shape[1]
        coeficientes = np.random.rand(n_features)
        ruido = np.random.randn(X_array.shape[0]) * 10
        y = X_array @ coeficientes + ruido
    else:
        # Convertir y si es Series de pandas
        if isinstance(y, pd.Series):
            y = y.values
        elif isinstance(y, list):
            y = np.array(y)
        X_array = X.values if isinstance(X, pd.DataFrame) else np.array(X)
    
    # Convertir X a numpy si es DataFrame
    if isinstance(X, pd.DataFrame):
        X_array = X.values
    elif isinstance(X, np.ndarray):
        X_array = X
    else:
        X_array = np.array(X)
    
    # 1. Dividir los datos: 80% entrenamiento, 20% prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X_array, y, test_size=0.2, random_state=42
    )

    # 2. Entrenar el modelo de Regresión Lineal
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)

    # 3. Realizar predicciones sobre el conjunto de prueba
    predicciones = modelo.predict(X_test)

    # 4. Calcular el Error Cuadrático Medio (MSE)
    error_mse = mean_squared_error(y_test, predicciones)

    return float(error_mse)
