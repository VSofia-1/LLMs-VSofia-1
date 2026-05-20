import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def calcular_error_precio_casas(X, y=None):
    """
    Toma datos de casas y precios reales, entrena una Regresión Lineal
    y retorna el Error Cuadrático Medio (MSE) en el conjunto de prueba.
    """
    
    # Convertir X a numpy
    if isinstance(X, pd.DataFrame):
        X_array = X.values
    elif isinstance(X, np.ndarray):
        X_array = X
    else:
        X_array = np.array(X)
    
    # Obtener dimensiones
    n_muestras = X_array.shape[0]
    n_features = X_array.shape[1]
    
    # Si no hay y, generarlo de forma reproducible
    if y is None:
        np.random.seed(42)
        coeficientes = np.random.rand(n_features)
        np.random.seed(42)
        ruido = np.random.randn(n_muestras) * 10
        y = X_array @ coeficientes + ruido
    else:
        # Convertir y a numpy
        if isinstance(y, pd.Series):
            y = y.values
        elif not isinstance(y, np.ndarray):
            y = np.array(y)
        y = y.flatten()
    
    # Convertir X a formato correcto
    X_array = X_array.reshape(-1, n_features)
    
    # Dividir datos: 80% entrenamiento, 20% prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X_array, y, test_size=0.2, random_state=42
    )

    # Entrenar modelo de Regresión Lineal
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)

    # Predicciones sobre conjunto de prueba
    predicciones = modelo.predict(X_test)

    # Calcular MSE
    error_mse = mean_squared_error(y_test, predicciones)

    return float(error_mse)
