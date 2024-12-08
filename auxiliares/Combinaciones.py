import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.preprocessing import MinMaxScaler

def graficar_combinaciones_3d_normalizado(csv_path, columnas):
    """
    Genera gráficos 3D para todas las combinaciones únicas de tres variables, con valores normalizados.

    Args:
        csv_path (str): Ruta al archivo CSV con los datos.
        columnas (list): Lista con los nombres de las columnas disponibles en el CSV.
    """
    try:
        # Cargar datos
        df = pd.read_csv(csv_path)

        # Normalizar las columnas seleccionadas
        scaler = MinMaxScaler()
        datos_normalizados = scaler.fit_transform(df.iloc[:, :-1])  # Excluir última columna (etiquetas)

        # Generar todas las combinaciones únicas de tres columnas
        combinaciones = list(combinations(range(len(columnas)), 3))

        for indices in combinaciones:
            var_x, var_y, var_z = indices

            # Nombres de las variables
            nombre_x, nombre_y, nombre_z = columnas[var_x], columnas[var_y], columnas[var_z]

            # Datos normalizados de las variables seleccionadas
            x = datos_normalizados[:, var_x]
            y = datos_normalizados[:, var_y]
            z = datos_normalizados[:, var_z]
            etiquetas = df.iloc[:, -1]  # Última columna como etiquetas

            # Crear el gráfico
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            categorias = np.unique(etiquetas)
            colores = ['r', 'g', 'b', 'y', 'c', 'm', 'k']

            for categoria, color in zip(categorias, colores):
                indices_categoria = etiquetas == categoria
                ax.scatter(x[indices_categoria], y[indices_categoria], z[indices_categoria], label=categoria, c=color)

            ax.set_title(f"Gráfico 3D (Normalizado): {nombre_x}, {nombre_y}, {nombre_z}")
            ax.set_xlabel(nombre_x)
            ax.set_ylabel(nombre_y)
            ax.set_zlabel(nombre_z)
            ax.legend()

            plt.show()

    except Exception as e:
        print(f"Error al generar gráficos: {e}")

# Configuración inicial
csv_path = "./DB/parametros_DB.csv"  # Ruta al archivo CSV
columnas = [
    "ZCR", "RMS", "Centroide", "Ancho_Banda", "MFCC_1", "MFCC_2", "MFCC_3", "MFCC_4",
    "MFCC_5", "MFCC_6", "MFCC_7", "MFCC_8", "MFCC_9", "MFCC_10", "MFCC_11", "MFCC_12",
    "MFCC_13"
]

# Llamar a la función
graficar_combinaciones_3d_normalizado(csv_path, columnas)
