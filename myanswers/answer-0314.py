import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
  
 
# ---------------------------------------------------------
# 2. FUNCIÓN PRINCIPAL
# ---------------------------------------------------------
def reducir_perfiles_categoricos(df, cat_cols, n_components):
 
    X_encoded = pd.get_dummies(df[cat_cols], columns=cat_cols)
 
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    X_reducida = svd.fit_transform(X_encoded)
 
    varianza_acumulada = float(np.sum(svd.explained_variance_ratio_))
 
    return X_reducida, varianza_acumulada
 
