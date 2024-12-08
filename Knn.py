import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os

class KNN:
    def __init__(self, archivo_parametros_db: str, archivo_parametros_candidato: str, k: int = 8):
        """
        Clase para clasificar un audio candidato utilizando KNN.

        Args:
            archivo_parametros_db (str): Ruta al archivo CSV con los parámetros de la base de datos.
            archivo_parametros_candidato (str): Ruta al archivo CSV con los parámetros del audio candidato.
            k (int): Número de vecinos a considerar en el algoritmo KNN (default: 3).
        """
        self.archivo_parametros_db = archivo_parametros_db
        self.archivo_parametros_candidato = archivo_parametros_candidato
        self.k = k
        self.variables_seleccionadas = [0, 1, 2]  # Por defecto, ZCR, RMS, Centroide Espectral



    def cargar_datos(self):
        """
        Carga y escala los datos de la base de datos y del audio candidato, utilizando solo las variables seleccionadas.

        Returns:
            X_train (np.array): Características de la base de datos filtradas y escaladas.
            y_train (np.array): Etiquetas de la base de datos.
            X_test (np.array): Características del audio candidato filtradas y escaladas.
        """
        try:
            # Cargar base de datos
            df_db = pd.read_csv(self.archivo_parametros_db)
            X_train = df_db.iloc[:, self.variables_seleccionadas].values  # Filtrar características
            y_train = df_db.iloc[:, -1].values  # Etiquetas

            # Cargar audio candidato
            df_candidato = pd.read_csv(self.archivo_parametros_candidato)
            X_test = df_candidato.iloc[:, self.variables_seleccionadas].values  # Filtrar características

            # Escalar datos
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            return X_train, y_train, X_test
        except Exception as e:
            print(f"Error al cargar los datos: {e}")
            return None, None, None


    def clasificar(self, usar_pca=False):
        """
        Clasifica el audio candidato utilizando KNN.

        Args:
            usar_pca (bool): Si se utiliza PCA para reducir las características a 3 componentes principales.

        Returns:
            str: Etiqueta predicha para el audio candidato.
        """
        X_train, y_train, X_test = self.cargar_datos()
        if X_train is None or y_train is None or X_test is None:
            print("Error al cargar los datos. No se puede clasificar.")
            return

        if usar_pca:
            pca = PCA(n_components=3)
            X_train = pca.fit_transform(X_train)
            X_test = pca.transform(X_test)

            # Gráfico 3D
            self.graficar_3d(X_train, y_train, X_test)

        # Entrenar modelo KNN
        knn = KNeighborsClassifier(n_neighbors=self.k, weights='uniform')
        knn.fit(X_train, y_train)

        # Predicción
        prediccion = knn.predict(X_test)
        print(f"El audio candidato fue clasificado como: {prediccion[0]}")
        return prediccion[0]

    def seleccionar_variables(self, variables):
        """
        Selecciona las variables del CSV que se utilizarán para la clasificación y graficación.

        Args:
            variables (list): Lista de índices de las columnas seleccionadas (deben ser exactamente 3).
        """
        if len(variables) != 3:
            raise ValueError("Debe seleccionar exactamente tres variables para graficar en 3D.")
        self.variables_seleccionadas = variables
        print(f"Variables seleccionadas para la clasificación: {self.variables_seleccionadas}")

    def graficar_3d(self, X_train, y_train, X_test):
        """
        Genera un gráfico 3D con las variables seleccionadas y el audio candidato.

        Args:
            X_train (np.array): Características de la base de datos escaladas.
            y_train (np.array): Etiquetas de la base de datos.
            X_test (np.array): Características del audio candidato escaladas.
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Graficar la base de datos
        etiquetas_unicas = np.unique(y_train)
        colores = ['r', 'g', 'b', 'y']
        for etiqueta, color in zip(etiquetas_unicas, colores):
            indices = np.where(y_train == etiqueta)
            ax.scatter(X_train[indices, 0], X_train[indices, 1], X_train[indices, 2], label=etiqueta, c=color)

        # Graficar el audio candidato
        ax.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2], label="Candidato", c='k', s=100, marker='*')

        # Etiquetas de ejes
        ax.set_title("Visualización en 3D (Variables Seleccionadas)")
        ax.set_xlabel(f"Variable {self.variables_seleccionadas[0]}")
        ax.set_ylabel(f"Variable {self.variables_seleccionadas[1]}")
        ax.set_zlabel(f"Variable {self.variables_seleccionadas[2]}")
        ax.legend()
        plt.show()

    def knn_layered(self):
        """
        Clasifica un audio candidato utilizando un enfoque de dos capas.
        - Primera capa: utiliza las variables 2, 3 y 7.
        - Segunda capa (si aplica): utiliza las variables 4, 12 y 19 para refinar la clasificación
        entre camote y zanahoria.

        Grafica los puntos correspondientes en cada capa e imprime los nombres de las columnas seleccionadas.
        """
        try:
            # Cargar datos
            df_db = pd.read_csv(self.archivo_parametros_db)
            df_candidato = pd.read_csv(self.archivo_parametros_candidato)

            # Obtener encabezados de las columnas
            headers = df_db.columns

            # Combinar datos para normalizar
            combined_data = pd.concat([df_db.iloc[:, :-1], df_candidato], ignore_index=True)
            scaler = StandardScaler()
            combined_data_normalized = scaler.fit_transform(combined_data)

            # Dividir los datos normalizados
            X_train_normalized = combined_data_normalized[:-1]
            X_test_normalized = combined_data_normalized[-1].reshape(1, -1)
            etiquetas = df_db.iloc[:, -1].values

            # Primera capa: variables [2, 3, 7]
            variables_primera_capa = [2, 3, 7]
            X_train_primera = X_train_normalized[:, variables_primera_capa]
            X_test_primera = X_test_normalized[:, variables_primera_capa]

            # Graficar la primera capa
            self.graficar_3d_layered(X_train_primera, etiquetas, X_test_primera, variables_primera_capa, "Primera Capa")

            # Entrenar KNN para la primera capa
            knn = KNeighborsClassifier(n_neighbors=self.k, weights='uniform')
            knn.fit(X_train_primera, etiquetas)
            prediccion_primera_capa = knn.predict(X_test_primera)[0]

            print(f"Predicción en la primera capa: {prediccion_primera_capa}")

            # Si no es camote o zanahoria, finalizar
            if prediccion_primera_capa not in ["camote", "zanahoria"]:
                print(f"El audio candidato fue clasificado como: {prediccion_primera_capa}")
                return prediccion_primera_capa

            # Segunda capa: variables [11, 18, 20] para camote y zanahoria
            variables_segunda_capa = [11, 18, 20]
            indices_filtrados = np.where((etiquetas == "camote") | (etiquetas == "zanahoria"))[0]
            X_train_segunda = X_train_normalized[indices_filtrados][:, variables_segunda_capa]
            y_train_segunda = etiquetas[indices_filtrados]
            X_test_segunda = X_test_normalized[:, variables_segunda_capa]

            # Obtener y mostrar los nombres de las columnas para la segunda capa
            columnas_segunda_capa = [headers[i] for i in variables_segunda_capa]
            print(f"Variables de la segunda capa: {columnas_segunda_capa}")

            # Graficar la segunda capa
            self.graficar_3d_layered(X_train_segunda, y_train_segunda, X_test_segunda, variables_segunda_capa, "Segunda Capa")

            # Entrenar KNN para la segunda capa
            knn.fit(X_train_segunda, y_train_segunda)
            prediccion_segunda_capa = knn.predict(X_test_segunda)[0]

            print(f"Predicción en la segunda capa: {prediccion_segunda_capa}")
            return prediccion_segunda_capa

        except Exception as e:
            print(f"Error en knn_layered: {e}")
            return None

        
    def graficar_3d_layered(self, X_train, y_train, X_test, variables, titulo):
        """
        Genera un gráfico 3D con los puntos de entrenamiento y el candidato.

        Args:
            X_train (np.array): Características de la base de datos filtradas.
            y_train (np.array): Etiquetas de la base de datos.
            X_test (np.array): Características del audio candidato filtradas.
            variables (list): Índices de las variables utilizadas en esta capa.
            titulo (str): Título del gráfico.
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Graficar la base de datos
        etiquetas_unicas = np.unique(y_train)
        colores = ['r', 'g', 'b', 'y']
        for etiqueta, color in zip(etiquetas_unicas, colores):
            indices = np.where(y_train == etiqueta)
            ax.scatter(X_train[indices, 0], X_train[indices, 1], X_train[indices, 2], label=etiqueta, c=color)

        # Graficar el audio candidato
        ax.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2], label="Candidato", c='k', s=100, marker='*')

        # Etiquetas de los ejes
        ax.set_title(f"{titulo} (Variables: {variables})")
        ax.set_xlabel(f"Variable {variables[0]}")
        ax.set_ylabel(f"Variable {variables[1]}")
        ax.set_zlabel(f"Variable {variables[2]}")
        ax.legend()
        plt.show()

    





# Función main para pruebas unitarias
def main():
    """
    Prueba unitaria de la clase KNN.
    - Clasifica el audio candidato utilizando las variables seleccionadas y grafica los resultados.
    """
    archivo_parametros_db = os.path.join(os.getcwd(), "DB", "parametros_DB.csv")
    archivo_parametros_candidato = os.path.join(os.getcwd(), "DB", "parametros_candidato.csv")

    knn = KNN(archivo_parametros_db, archivo_parametros_candidato, k=3)

    print("\n--- Selección de Variables Manual ---")
    # Seleccionar variables manualmente (por índices)
    knn.seleccionar_variables([2, 3, 7])  # Ejemplo: Centroide, Ancho de Banda, MFCC_2

    print("\nClasificación con variables seleccionadas:")
    knn.clasificar()

    print("\nClasificación con PCA:")
    knn.clasificar(usar_pca=True)


if __name__ == "__main__":
    main()
