import numpy as np
import random
from sklearn.metrics.pairwise import rbf_kernel

# ---------------- FUNCIÓN PRINCIPAL ----------------

def calcular_kernel_rbf(X, gamma):
    kernel_matrix = rbf_kernel(X, gamma=gamma)    

    return rbf_kernel(X, gamma=gamma)

input_case, output_case = generar_caso_de_uso_kernel()

# usar tu función
resultado = calcular_kernel_rbf(
    input_case["X"],
    input_case["gamma"]
)

print("INPUT:")
print(input_case)

print("\nOUTPUT")
print(resultado)
