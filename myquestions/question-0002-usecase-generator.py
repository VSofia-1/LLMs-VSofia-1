import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
import random

def generar_caso_de_uso_curva_roc():
    
    # número aleatorio de muestras
    n_samples = random.randint(20, 20)
    
    # generar etiquetas reales binarias
    y_true = np.random.randint(0, 2, size=n_samples)
    
    # asegurar que haya al menos una clase positiva y una negativa
    if len(np.unique(y_true)) < 2:
        y_true[0] = 0
        y_true[1] = 1
    
    # generar scores aleatorios (probabilidades)
    y_scores = np.random.rand(n_samples)
    
    # calcular valores esperados
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores)
    
    input = {
        "y_true": y_true,
        "y_scores": y_scores
    }
    
    output = (fpr, tpr, auc)
    
    return input, output
input_case, output_case = generar_caso_de_uso_curva_roc()

print("INPUT:")
print(input_case)

print("\nOUTPUT:")
print(output_case)
