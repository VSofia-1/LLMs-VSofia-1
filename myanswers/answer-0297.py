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
# CASO DE USO
# ---------------------------------------------------------
def generar_caso_de_uso_calcular_error_precio_casas():

    n_muestras = np.random.randint(50, 200)
    n_features = np.random.randint(2, 5)

    X = np.random.rand(n_muestras, n_features) * 100

    coeficientes = np.random.rand(n_features)

    ruido = np.random.randn(n_muestras) * 10

    y = X @ coeficientes + ruido

    X_df = pd.DataFrame(X)

    info = {
        "descripcion":
        "Cálculo de error en predicción de precios de casas",

        "n_muestras": n_muestras,

        "n_features": n_features
    }

    return info, X_df, y


# ---------------------------------------------------------
# EJECUCIÓN
# ---------------------------------------------------------
if __name__ == "__main__":

    info, X, y = (
        generar_caso_de_uso_calcular_error_precio_casas()
    )

    print("=== INFORMACIÓN ===")
    print(info)

    print("\n=== DATOS X ===")
    print(X.head())

    print("\n=== DATOS y ===")
    print(y[:5])

    resultado = calcular_error_precio_casas(X, y)

    print("\n=== MSE DEL MODELO ===")
    print(resultado)