import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
  
 
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
 
 
# ---------------------------------------------------------
# 1. GENERADOR DE CASO DE USO (GROUND TRUTH)
# ---------------------------------------------------------
def generar_caso_de_uso_reducir_perfiles_categoricos():
    rng = np.random.default_rng()
 
    n_filas = int(rng.integers(40, 90))
 
    ciudades  = ["Bogota", "Medellin", "Cali", "Barranquilla", "Cartagena",
                 "Bucaramanga", "Manizales", "Pereira", "Tunja", "Pasto"]
    planes    = ["Basico", "Estandar", "Premium", "Empresarial", "Familiar", "Plus"]
    canales   = ["Web", "App", "Tienda", "CallCenter", "Distribuidor"]
    segmentos = ["Joven", "Adulto", "Pyme", "Corporativo", "Estudiante", "Hogar"]
 
    df = pd.DataFrame({
        "ciudad":       rng.choice(ciudades,  size=n_filas),
        "tipo_plan":    rng.choice(planes,    size=n_filas),
        "canal_compra": rng.choice(canales,   size=n_filas),
        "segmento":     rng.choice(segmentos, size=n_filas)
    })
    df["antiguedad_meses"] = rng.integers(1, 60, size=n_filas)
 
    cat_cols = ["ciudad", "tipo_plan", "canal_compra", "segmento"]
 
    X_encoded = pd.get_dummies(df[cat_cols], columns=cat_cols)
    max_componentes = min(X_encoded.shape[0], X_encoded.shape[1]) - 1
 
    if max_componentes < 2:
        n_components = 1
    else:
        n_components = int(rng.integers(2, min(6, max_componentes) + 1))
 
    input_data = {
        "df":           df,
        "cat_cols":     cat_cols,
        "n_components": n_components
    }
 
    output_data = reducir_perfiles_categoricos(df, cat_cols, n_components)
 
    return input_data, output_data
 
 
# ---------------------------------------------------------
# 2. FUNCIÓN PRINCIPAL
# ---------------------------------------------------------
def reducir_perfiles_categoricos(df, cat_cols, n_components):
 
    X_encoded = pd.get_dummies(df[cat_cols], columns=cat_cols)
 
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    X_reducida = svd.fit_transform(X_encoded)
 
    varianza_acumulada = float(np.sum(svd.explained_variance_ratio_))
 
    return X_reducida, varianza_acumulada
 
 
# ---------------------------------------------------------
# 3. COMPARACIÓN
# ---------------------------------------------------------
if __name__ == "__main__":
    import math
 
    N = 10
    correctos = 0
 
    print(f"{'='*65}")
    print(f"  Comparación: esperado (generador)  vs  obtenido (función)")
    print(f"{'='*65}\n")
 
    for i in range(1, N + 1):
        entrada, (X_esp, var_esp) = generar_caso_de_uso_reducir_perfiles_categoricos()
 
        try:
            X_obt, var_obt = reducir_perfiles_categoricos(
                entrada["df"],
                entrada["cat_cols"],
                entrada["n_components"]
            )
 
            shapes_ok  = X_esp.shape == X_obt.shape
            arrays_ok  = np.allclose(X_esp, X_obt, rtol=1e-5)
            varianza_ok = math.isclose(var_esp, var_obt, rel_tol=1e-5)
 
            if shapes_ok and arrays_ok and varianza_ok:
                print(f"Test {i:02d} | n_components={entrada['n_components']}"
                      f" | shape={X_obt.shape} | varianza={var_obt:.4f} | ✓ CORRECTO")
                correctos += 1
            else:
                print(f"Test {i:02d} | ✗ INCORRECTO"
                      f" | shapes_ok={shapes_ok} arrays_ok={arrays_ok} varianza_ok={varianza_ok}")
 
        except Exception as e:
            print(f"Test {i:02d} | ✗ ERROR DE EJECUCIÓN: {e}")
 
    print(f"\n{'─'*65}")
    print(f"  Resultado: {correctos}/{N} correctos  |  {N - correctos}/{N} incorrectos")
    print(f"{'─'*65}")
