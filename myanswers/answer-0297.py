import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


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
        "descripcion": "Cálculo de error en predicción de precios de casas",
        "n_muestras": n_muestras,
        "n_features": n_features
    }

    # IMPORTANTE:
    return info, X_df, y


# ---------------------------------------------------------
# FUNCIÓN PRINCIPAL
# ---------------------------------------------------------
def calcular_error_precio_casas(info, X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    modelo = LinearRegression()

    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)

    return float(mse)
