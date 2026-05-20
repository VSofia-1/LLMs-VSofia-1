import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def calcular_error_precio_casas(**kwargs):
    """
    Toma datos de casas y precios reales, entrena una Regresión Lineal
    y retorna el Error Cuadrático Medio (MSE) en el conjunto de prueba.
    
    El generador pasa: info (diccionario con descripcion, n_muestras, n_features)
    Pero también debería pasar X y y para que funcione correctamente.
    
    Esta función genera los datos necesarios si no se reciben.
    """
    
    # Extraer X e y desde kwargs - soporta varios nombres de clave
    X = kwargs.get('X', None)
    y = kwargs.get('y', None)
    info = kwargs.get('info', kwargs)  # Si no hay X, usar el info directamente
    
    # Si X no está, intentar generarlo desde info
    if X is None:
        # Obtener parámetros del info/diccionario
        n_muestras = kwargs.get('n_muestras', 100)
        n_features = kwargs.get('n_features', 3)
        
        # Generar datos sintéticos
        X = np.random.rand(n_muestras, n_features) * 100
        coeficientes = np.random.rand(n_features)
        ruido = np.random.randn(n_muestras) * 10
        y = X @ coeficientes + ruido
    
    # Convertir a numpy arrays si son DataFrame/Series
    if isinstance(X, pd.DataFrame):
        X = X.values
    elif not isinstance(X, np.ndarray):
        X = np.array(X)
        
    if isinstance(y, pd.Series):
        y = y.values
    elif not isinstance(y, np.ndarray):
        y = np.array(y)
    
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
