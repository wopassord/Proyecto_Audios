import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


def knn_layered_iterativo(csv_path, k_max=20):
    """
    Itera sobre los audios en el CSV para encontrar el mejor valor de k en el algoritmo KNN Layered.

    Args:
        csv_path (str): Ruta al archivo CSV con los datos parametrizados.
        k_max (int): Máximo valor de k a evaluar.

    Returns:
        int: Mejor valor de k basado en la precisión de clasificación.
    """
    try:
        # Cargar datos
        df = pd.read_csv(csv_path)
        X = df.iloc[:, :-1].values  # Todas las columnas excepto la última son variables
        y = df.iloc[:, -1].values   # Última columna contiene las etiquetas

        # Normalizar las variables
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        total_audios = len(X)
        resultados_k = np.zeros(k_max)

        # Iterar sobre cada audio como candidato
        for i in range(total_audios):
            # Separar el audio candidato y la base de datos
            X_candidato = X[i].reshape(1, -1)
            y_candidato = y[i]
            X_base = np.delete(X, i, axis=0)
            y_base = np.delete(y, i, axis=0)

            # Evaluar KNN Layered para cada valor de k
            for k in range(1, k_max + 1):
                # --- Primera Capa ---
                # Usar variables [2, 3, 7] (Centroide, Ancho de Banda, MFCC_3)
                variables_primera_capa = [2, 3, 7]
                X_train_primera = X_base[:, variables_primera_capa]
                X_test_primera = X_candidato[:, variables_primera_capa]

                knn = KNeighborsClassifier(n_neighbors=k, weights='uniform')
                knn.fit(X_train_primera, y_base)
                prediccion_primera_capa = knn.predict(X_test_primera)[0]

                # Si la predicción no es camote o zanahoria, finalizar esta iteración
                if prediccion_primera_capa not in ["camote", "zanahoria"]:
                    if prediccion_primera_capa == y_candidato:
                        resultados_k[k - 1] += 1
                    continue

                # --- Segunda Capa ---
                # Usar variables [4, 12, 19] (MFCC_1, MFCC_9, MFCC_16)
                variables_segunda_capa = [11, 18, 20]
                indices_filtrados = np.where((y_base == "camote") | (y_base == "zanahoria"))[0]
                X_train_segunda = X_base[indices_filtrados][:, variables_segunda_capa]
                y_train_segunda = y_base[indices_filtrados]
                X_test_segunda = X_candidato[:, variables_segunda_capa]

                knn.fit(X_train_segunda, y_train_segunda)
                prediccion_segunda_capa = knn.predict(X_test_segunda)[0]

                # Verificar predicción final
                if prediccion_segunda_capa == y_candidato:
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


# Configuración inicial
csv_path = "./DB/parametros_DB.csv"  # Ruta al archivo CSV
k_max = 20  # Máximo valor de k a evaluar

# Ejecutar el algoritmo
mejor_k = knn_layered_iterativo(csv_path, k_max)
