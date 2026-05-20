import numpy as np
import pandas as pd
import math
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
 
 
# ── Función principal ─────────────────────────────────────────────────────────
def calcular_error_precio_casas(X, y) -> float:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)
    return float(mean_squared_error(y_test, modelo.predict(X_test)))

resultado = calcular_error_precio_casas(X_df, y)
resultado
