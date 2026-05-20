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
 
 
# ── Generador de casos de uso ─────────────────────────────────────────────────
def generar_caso_de_uso(seed):
    np.random.seed(seed)
    n_muestras = np.random.randint(50, 200)
    n_features = np.random.randint(2, 5)
 
    X = np.random.rand(n_muestras, n_features) * 100
    coeficientes = np.random.rand(n_features)
    ruido = np.random.randn(n_muestras) * 10
    y = X @ coeficientes + ruido
 
    return pd.DataFrame(X), y
 
 
# ── MSE de referencia (esperado) ──────────────────────────────────────────────
def mse_esperado(X_df, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y, test_size=0.2, random_state=42
    )
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)
    return float(mean_squared_error(y_test, modelo.predict(X_test)))
 
 
# ── Comparación ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    N = 10
    correctos = 0
 
    print(f"{'='*72}")
    print(f"  Comparación: esperado (generador)  vs  obtenido (función principal)")
    print(f"{'='*72}\n")
 
    for i in range(1, N + 1):
        X_df, y = generar_caso_de_uso(seed=i)
 
        esperado = mse_esperado(X_df, y)
        obtenido = calcular_error_precio_casas(X=X_df, y=y)
 
        match  = math.isclose(esperado, obtenido, rel_tol=1e-5)
        estado = "✓ CORRECTO  " if match else "✗ INCORRECTO"
        correctos += match
 
        print(f"Test {i:02d} | muestras={len(X_df):>3}  features={X_df.shape[1]}"
              f" | esperado={esperado:>12.4f}  obtenido={obtenido:>12.4f}  | {estado}")
        if not match:
            print(f"         └─ diferencia: {abs(esperado - obtenido):.6f}")
 
    print(f"\n{'─'*72}")
    print(f"  Resultado: {correctos}/{N} correctos  |  {N - correctos}/{N} incorrectos")
    print(f"{'─'*72}")
