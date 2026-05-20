import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def calcular_error_precio_casas(X, y):
    """
    Toma datos de casas y precios reales, entrena una Regresión Lineal
    y retorna el Error Cuadrático Medio (MSE) en el conjunto de prueba.

    Args:
        X (pd.DataFrame o np.ndarray): Variables predictoras (m2, habitaciones, etc.).
        y (pd.Series o np.ndarray): Precio real de las casas.

    Returns:
        float: Error Cuadrático Medio (MSE) del modelo.
    """
    
    # 1. Dividir los datos: 80% entrenamiento, 20% prueba
    # Usamos random_state para reproducibilidad
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

# --- BLOQUE PARA PROBAR LA FUNCIÓN (Usando el generador de ejemplo) ---

# Aquí simulamos el caso de uso que、提供iste para generar datos de prueba
def probar_funcion():
    # Generamos los datos usando la lógica de tu ejemplo
    n_muestras = np.random.randint(50, 200)
    n_features = np.random.randint(2, 5)

    X = np.random.rand(n_muestras, n_features) * 100
    # Generamos 'y' con la misma fórmula para tener una correlación lógica
    coeficientes = np.random.rand(n_features)
    ruido = np.random.randn(n_muestras) * 10
    y = X @ coeficientes + ruido

    # Convertimos a DataFrame/Series si es necesario para tu validación
    X_df = pd.DataFrame(X)
    y_series = pd.Series(y)

    # Llamamos a nuestra función
    resultado = calcular_error_precio_casas(X_df, y_series)
    
    print(f"El Error Cuadrático Medio (MSE) calculado es: {resultado}")
    return resultado

# Descomenta la siguiente línea para probar el código:
# probar_funcion()
