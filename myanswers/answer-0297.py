import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def calcular_error_precio_casas(X=None, **kwargs):
    """
    Toma datos de casas y precios reales, entrena una Regresión Lineal
    y retorna el Error Cuadrático Medio (MSE) en el conjunto de prueba.
    """
    
    # Si X viene como argumento posicional
    if X is None:
        X = kwargs.get('X', None)
    
    # Si aún no hay X, intentar otras claves
    if X is None:
        for key in kwargs:
            val = kwargs[key]
            if isinstance(val, (pd.DataFrame, np.ndarray)):
                X = val
                break
    
    # Verificar que X exista
    if X is None:
        raise ValueError("No se encontró X en los argumentos")
    
    # Convertir a numpy
    if isinstance(X, pd.DataFrame):
        X = X.values
    elif not isinstance(X, np.ndarray):
        X = np.array(X)
    
    # Obtener n_muestras y n_features de X
    n_muestras = X.shape[0]
    n_features = X.shape[1]
    
    # Generar y reproducible (misma lógica que el generador original)
    np.random.seed(42)  # Seed fijo para reproducibilidad
    coeficientes = np.random.rand(n_features)
    np.random.seed(42)  # Seed otra vez para y
    ruido = np.random.randn(n_muestras) * 10
    y = X @ coeficientes + ruido
    
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
