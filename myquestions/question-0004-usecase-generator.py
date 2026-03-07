import numpy as np
import random
from sklearn.metrics.pairwise import rbf_kernel

def generar_caso_de_uso_kernel():
    
    # número aleatorio de muestras y características
    n_samples = random.randint(5, 10)
    n_features = random.randint(2, 3)
    
    # generar matriz X aleatoria
    X = np.random.randn(n_samples, n_features)
    
    # gamma aleatorio
    gamma = random.uniform(0.1, 2.0)
    
    # calcular kernel esperado
    kernel_matrix = rbf_kernel(X, gamma=gamma)
    
    input = {
        "X": X,
        "gamma": gamma
    }
    
    output = kernel_matrix
    
    return input, output

input_case, output_case = generar_caso_de_uso_kernel()

print("INPUT:")
print(input_case)

print("\nOUTPUT:")
print(output_case)
