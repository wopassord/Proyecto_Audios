import pandas as pd
import numpy as np
from itertools import combinations
from scipy.spatial.distance import cdist

def encontrar_mejor_terna(csv_path):
    """
    Encuentra la terna de parámetros que maximiza la distancia euclideana promedio entre categorías.

    Args:
        csv_path (str): Ruta al archivo CSV con los parámetros y etiquetas.

    Returns:
        tuple: La mejor terna de parámetros y su distancia promedio.
    """
    try:
        # Cargar datos
        df = pd.read_csv(csv_path)
        columnas = df.columns[:-1]  # Excluir la última columna (etiquetas)
        etiquetas = df.columns[-1]  # Última columna (etiquetas)
        
        # Agrupar datos por categoría
        grupos = df.groupby(etiquetas)
        
        # Generar todas las combinaciones de tres parámetros
        combinaciones_parametros = list(combinations(columnas, 3))
        mejor_terna = None
        max_distancia_promedio = 0
        
        # Evaluar cada terna
        for terna in combinaciones_parametros:
            distancias_totales = []
            # Extraer los datos de las columnas seleccionadas
            for categoria_a in grupos.groups:
                puntos_a = df.loc[grupos.groups[categoria_a], list(terna)].values
                for categoria_b in grupos.groups:
                    if categoria_a != categoria_b:
                        puntos_b = df.loc[grupos.groups[categoria_b], list(terna)].values
                        # Calcular distancias euclideanas entre todas las combinaciones de puntos
                        distancias = cdist(puntos_a, puntos_b, metric='euclidean')
                        distancias_totales.append(np.mean(distancias))
            
            # Calcular la distancia promedio para esta terna
            distancia_promedio = np.mean(distancias_totales)
            
            # Actualizar la mejor terna si esta combinación supera la distancia máxima encontrada
            if distancia_promedio > max_distancia_promedio:
                max_distancia_promedio = distancia_promedio
                mejor_terna = terna
        
        return mejor_terna, max_distancia_promedio

    except Exception as e:
        print(f"Error al procesar el archivo CSV: {e}")
        return None, None

# Ruta al archivo "parametros_DB.csv"
csv_path = "C:\\Users\\berni\\Desktop\\ProyectoIAPrima\\DB\parametros_DB.csv"

# Encontrar la mejor terna de parámetros
mejor_terna, distancia = encontrar_mejor_terna(csv_path)

# Mostrar los resultados
if mejor_terna is not None:
    print(f"\nMejor terna de parámetros: {mejor_terna}")
    print(f"Distancia promedio máxima entre categorías: {distancia:.2f}")
