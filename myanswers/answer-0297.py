import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def calcular_error_precio_casas(**kwargs):
    """
    Toma datos de casas y precios reales, entrena una Regresión Lineal
    y retorna el Error Cuadrático Medio (MSE).
    """
    
    # Intentar obtener X e y de kwargs
    X = kwargs.get('X', None)
    y = kwargs.get('y', None)
    
    # Si no hay X, generar datos desde los parámetros del info
    if X is None:
        n_muestras = kwargs.get('n_muestras', 100)
        n_features = kwargs.get('n_features', 3)
        
        np.random.seed(42)  # Para reproducibilidad
        X = np.random.rand(n_muestras, n_features) * 100
        coeficientes = np.random.rand(n_features)
        ruido = np.random.randn(n_muestras) * 10
        y = X @ coeficientes + ruido
    
    # Convertir a numpy arrays
    if isinstance(X, pd.DataFrame):
        X = X.values
    elif not isinstance(X, np.ndarray):
        X = np.array(X)
        
    if isinstance(y, pd.Series):
        y = y.values
    elif not isinstance(y, np.ndarray):
        y = np.array(y)
    
    # Dividir datos: 80% entrenamiento, 20% prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Entrenar modelo de Regresión Lineal
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)

    # Predicciones sobre conjunto de prueba
    predicciones = modelo.predict(X_test)

    # Calcular MSE
    error_mse = mean_squared_error(y_test, predicciones)

    return float(error_mse)
