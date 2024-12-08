import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os

class KNN:
    def __init__(self, archivo_parametros_db: str, archivo_parametros_candidato: str, k: int = 10):
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
        Carga los datos de la base de datos y del audio candidato, utilizando solo las variables seleccionadas.

        Returns:
            X_train (np.array): Características de la base de datos filtradas.
            y_train (np.array): Etiquetas de la base de datos.
            X_test (np.array): Características del audio candidato filtradas.
        """
        try:
            # Cargar base de datos
            df_db = pd.read_csv(self.archivo_parametros_db)
            X_train = df_db.iloc[:, self.variables_seleccionadas].values  # Filtrar características
            y_train = df_db.iloc[:, -1].values  # Etiquetas

            # Cargar audio candidato
            df_candidato = pd.read_csv(self.archivo_parametros_candidato)
            X_test = df_candidato.iloc[:, self.variables_seleccionadas].values  # Filtrar características

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
        knn = KNeighborsClassifier(n_neighbors=self.k, weights='distance')
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
            X_train (np.array): Características de la base de datos filtradas.
            y_train (np.array): Etiquetas de la base de datos.
            X_test (np.array): Características del audio candidato filtradas.
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
