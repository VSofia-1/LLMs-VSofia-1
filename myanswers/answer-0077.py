import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal


# ---------------------------------------------------------
# 1. FUNCIÓN PRINCIPAL
# ---------------------------------------------------------
def analizar_umbrales_geotecnicos(df, umbral_nulos):

    df_limpio = df.copy()

    # 1. Eliminar columnas con muchos nulos
    porcentaje_nulos = df_limpio.isnull().mean()

    columnas_a_eliminar = porcentaje_nulos[
        porcentaje_nulos > umbral_nulos
    ].index

    df_limpio = df_limpio.drop(columns=columnas_a_eliminar)

    # 2. Imputación con moda
    columnas_numericas = df_limpio.select_dtypes(
        include=np.number
    ).columns

    for col in columnas_numericas:

        if df_limpio[col].isnull().any():

            if df_limpio[col].dropna().empty:
                continue

            moda = df_limpio[col].mode()[0]

            df_limpio[col] = df_limpio[col].fillna(moda)

    # 3. Codificación
    df_limpio['estabilidad_num'] = df_limpio['estabilidad'].map({
        'Inestable': 0,
        'Estable': 1
    })

    # 4. Resumen estadístico
    df_agrupado = df_limpio.groupby('estabilidad_num').agg({
        'pluviosidad_reciente': 'max',
        'presion_de_poros': 'mean'
    })

    # Ordenar índice y columnas
    df_agrupado = df_agrupado.sort_index().sort_index(axis=1)

    return df_agrupado


# ---------------------------------------------------------
# EJECUCIÓN Y VALIDACIÓN
# ---------------------------------------------------------
if __name__ == "__main__":

    entrada, salida_esperada = (
        generar_caso_de_uso_analisis_geotecnico()
    )

    print("=== INPUT ===")
    print(entrada['df'].head())

    print("\nUmbral:", entrada['umbral_nulos'])

    resultado = analizar_umbrales_geotecnicos(
        entrada['df'],
        entrada['umbral_nulos']
    )

    print("\n=== RESULTADO FUNCIÓN ===")
    print(resultado)
