import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
 
 
def calcular_error_precio_casas(descripcion: str, n_muestras: int, n_features: int) -> float:
    np.random.seed(42)
 
    X = np.random.rand(n_muestras, n_features) * 100
    coeficientes = np.random.rand(n_features)
    ruido = np.random.randn(n_muestras) * 10
    y = X @ coeficientes + ruido
 
    X_df = pd.DataFrame(X)
 
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y, test_size=0.2, random_state=42
    )
 
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)
 
    y_pred = modelo.predict(X_test)
 
    mse = mean_squared_error(y_test, y_pred)
 
    return float(mse)
