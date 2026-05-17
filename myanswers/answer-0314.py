import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD

# ---------------------------------------------------------
# 2. FUNCIÓN PRINCIPAL
# ---------------------------------------------------------
def reducir_perfiles_categoricos(
    df,
    cat_cols,
    n_components
):

    # One-Hot Encoding
    X_encoded = pd.get_dummies(df[cat_cols])

    # Reducción de dimensionalidad
    svd = TruncatedSVD(
        n_components=n_components,
        random_state=42
    )

    X_reducida = svd.fit_transform(X_encoded)

    # Varianza explicada acumulada
    varianza_acumulada = np.sum(
        svd.explained_variance_ratio_
    )

    return X_reducida, varianza_acumulada

# ---------------------------------------------------------
# 4. EJECUCIÓN Y VALIDACIÓN
# ---------------------------------------------------------
if __name__ == "__main__":

    entrada, salida_esperada = (
        generar_caso_de_uso_perfiles()
    )

    print("=== INPUT ===")
    print(entrada["df"].head())

    print("\nColumnas categóricas:")
    print(entrada["cat_cols"])

    print("\nNúmero de componentes:")
    print(entrada["n_components"])

    print("\n=== OUTPUT ESPERADO ===")

    print("\nMatriz Reducida:")
    print(salida_esperada[0])

    print("\nVarianza Acumulada:")
    print(salida_esperada[1])

    resultado = reducir_perfiles_categoricos(
        entrada["df"],
        entrada["cat_cols"],
        entrada["n_components"]
    )

    print("\n=== RESULTADO FUNCIÓN ===")

    print("\nMatriz Reducida:")
    print(resultado[0])

    print("\nVarianza Acumulada:")
    print(resultado[1])
