import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

def knn_iterativo(csv_path, variables, k_max=20):
    """
    Itera sobre los audios en el CSV para encontrar el mejor valor de k en el algoritmo KNN.

    Args:
        csv_path (str): Ruta al archivo CSV con los datos parametrizados.
        variables (list): Índices de las columnas del CSV a utilizar como variables para KNN.
        k_max (int): Máximo valor de k a evaluar.

    Returns:
        int: Mejor valor de k basado en la precisión de clasificación.
    """
    try:
        # Cargar datos
        df = pd.read_csv(csv_path)
        X = df.iloc[:, variables].values  # Seleccionar variables de interés
        y = df.iloc[:, -1].values         # Etiquetas (última columna)

        total_audios = len(X)
        resultados_k = np.zeros(k_max)

        # Iterar sobre cada audio como candidato
        for i in range(total_audios):
            # Separar el audio candidato y la base de datos
            X_candidato = X[i].reshape(1, -1)
            y_candidato = y[i]
            X_base = np.delete(X, i, axis=0)
            y_base = np.delete(y, i, axis=0)

            # Evaluar KNN para cada valor de k
            for k in range(1, k_max + 1):
                knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
                knn.fit(X_base, y_base)
                prediccion = knn.predict(X_candidato)

                # Incrementar el contador si la predicción es correcta
                if prediccion[0] == y_candidato:
                    resultados_k[k - 1] += 1

        # Calcular precisión para cada valor de k
        precisiones_k = resultados_k / total_audios

        # Encontrar el mejor k
        mejor_k = np.argmax(precisiones_k) + 1

        # Mostrar resultados
        print("\nResultados para cada valor de k:")
        for k in range(1, k_max + 1):
            print(f"k = {k}, Precisión = {precisiones_k[k - 1]:.2%}")

        print(f"\nEl mejor valor de k es: {mejor_k} con una precisión de {precisiones_k[mejor_k - 1]:.2%}")
        return mejor_k

    except Exception as e:
        print(f"Error: {e}")
        return None

# Parámetros
csv_path = "./DB/parametros_DB.csv"  # Ruta al CSV
variables_seleccionadas = [2, 3, 7]  # Índices de las variables seleccionadas (Centroide, Ancho Banda, MFCC_3)
k_max = 20  # Máximo valor de k a evaluar

# Ejecutar el algoritmo
mejor_k = knn_iterativo(csv_path, variables_seleccionadas, k_max)
