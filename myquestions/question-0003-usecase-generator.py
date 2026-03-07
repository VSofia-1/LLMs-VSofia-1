import numpy as np
import random
from sklearn.model_selection import KFold

def generar_caso_de_uso_splits_kflod():
    
    # número aleatorio de muestras y features
    n_samples = random.randint(5, 15)
    n_features = random.randint(2, 5)
    
    # generar dataset aleatorio
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 2, size=n_samples)
    
    # elegir k válido
    k = random.randint(2, min(6, n_samples))
    
    # random_state aleatorio
    random_state = random.randint(0, 30)
    
    # calcular los splits esperados
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
    
    splits = []
    for train_idx, test_idx in kf.split(X):
        splits.append((train_idx, test_idx))
    
    input = {
        "X": X,
        "y": y,
        "k": k,
        "random_state": random_state
    }
    
    output = splits
    
    return input, output
    
input_case, output_case = generar_caso_de_uso_splits_kflod()

print("INPUT:")
print(input_case)

print("\nOUTPUT:")
for i, (train_idx, test_idx) in enumerate(output_case):
    print(f"Split {i+1}")
    print("Train:", train_idx)
    print("Test:", test_idx)
