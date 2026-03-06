import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import random

def generar_caso_de_uso_reducir_dimensionalidad():
    n_samples = random.randint(10, 30)
    n_features = random.randint(2, 5)

    columnas = [f"feature_{i}" for i in range(n_features)]

    data = np.random.randn(n_samples, n_features)
    df = pd.DataFrame(data, columns=columnas)

    n_cols_usar = random.randint(2, n_features)
    columnas_seleccionadas = random.sample(columnas, n_cols_usar)

    random_state = random.randint(0, 100)

    X = df[columnas_seleccionadas].values
    X_scaled = StandardScaler().fit_transform(X)

    perplexity = min(5, n_samples - 1)

    tsne = TSNE(n_components=2, random_state=random_state, perplexity=perplexity)
    resultado = tsne.fit_transform(X_scaled)

    input_case = {
        "df": df,
        "columnas": columnas_seleccionadas,
        "random_state": random_state
    }

    output_case = resultado

    return input_case, output_case

input_case, output_case = generar_caso_de_uso_reducir_dimensionalidad()

print("INPUT:")
print(input_case)

print("\nOUTPUT:")
print(output_case)
