import os
from Preprocesador import Preprocesador
from Parametrizador import Parametrizador
from Grabadora import Grabadora
from Knn import KNN

class ClasificadorAudios:
    def __init__(self):
        """
        Clase principal que integra las funcionalidades de las demás clases.
        """
        # Definición de rutas
        self.carpeta_db = os.path.join(os.getcwd(), "DB")
        self.carpeta_crudos = os.path.join(self.carpeta_db, "Crudos")
        self.carpeta_procesados = os.path.join(self.carpeta_db, "Processed")
        self.carpeta_aumentados = os.path.join(self.carpeta_db, "Aumentados")
        self.carpeta_candidato = os.path.join(os.getcwd(), "Candidato")
        self.archivo_parametros_db = os.path.join(self.carpeta_db, "parametros_DB.csv")
        self.archivo_parametros_candidato = os.path.join(self.carpeta_db, "parametros_candidato.csv")
        self.preprocesador = Preprocesador(self.carpeta_crudos, self.carpeta_procesados)
        self.parametrizador_db = Parametrizador(self.carpeta_aumentados, self.archivo_parametros_db)
        self.parametrizador_candidato = Parametrizador(self.carpeta_candidato, self.archivo_parametros_candidato)
        self.grabadora = Grabadora(self.carpeta_candidato)
        self.knn = KNN(self.archivo_parametros_db, self.archivo_parametros_candidato, k=10)

    def mostrar_menu(self):
        """
        Muestra un menú interactivo para que el usuario seleccione opciones.
        """
        while True:
            print("\n--- Clasificador de Audios ---")
            print("1. Procesar Base de Datos")
            print("2. Extraer Parámetros de la Base de Datos")
            print("3. Grabar Audio Candidato")
            print("4. Extraer Parámetros del Audio Candidato")
            print("5. Clasificar el Audio Candidato")
            print("6. Clasificar con Variables Seleccionadas")
            print("7. Aumentar Datos de la Base de Datos")
            print("8. Grabar nuevo audio")
            print("9. Salir")
            opcion = input("Seleccione una opción: ")

            if opcion == "1":
                self.procesar_base_datos()
            elif opcion == "2":
                self.extraer_parametros_db()
            elif opcion == "3":
                self.grabar_audio_candidato()
            elif opcion == "4":
                self.extraer_parametros_candidato()
            elif opcion == "5":
                self.clasificar_audio_candidato()
            elif opcion == "6":
                self.clasificar_audio_con_variables()
            elif opcion == "7":
                self.aumentar_datos_db()
            elif opcion == '8':
                self.grabar_audio_crudo()
            elif opcion == "9":
                print("Saliendo del sistema. ¡Adiós!")
                break
            else:
                print("Opción inválida. Intente nuevamente.")

    def procesar_base_datos(self):
        """
        Ejecuta el preprocesamiento de los audios crudos en la base de datos.
        """
        print("Procesando base de datos...")
        self.preprocesador.procesar_audios()

    def extraer_parametros_db(self):
        """
        Extrae los parámetros de los audios de la base de datos.
        """
        print("Extrayendo parámetros de la base de datos...")
        self.parametrizador_db.extraer_parametros()

    def grabar_audio_candidato(self):
        """
        Permite grabar un audio candidato y procesarlo.
        """
        verdura = input("Ingrese el nombre de la verdura para la grabación (zanahoria, papa, camote, berenjena): ").strip().lower()
        self.grabadora.grabar_audio(verdura)
        self.grabadora.preprocesar_audio_candidato()

    def extraer_parametros_candidato(self):
        """
        Extrae los parámetros del audio candidato (solo el preprocesado).
        """
        print("Extrayendo parámetros del audio candidato...")
        archivo_candidato = os.path.join(self.carpeta_candidato, "processed_audio_candidato.wav")
        if not os.path.exists(archivo_candidato):
            print("No se encontró el archivo processed_audio_candidato.wav. Asegúrese de preprocesar el audio primero.")
            return
        self.parametrizador_candidato.parametrizar_candidato()

    def clasificar_audio_candidato(self):
        """
        Clasifica el audio candidato utilizando KNN, con y sin PCA.
        """
        print("\n--- Clasificación del Audio Candidato ---")
        print("1. Clasificar sin PCA")
        print("2. Clasificar con PCA")
        opcion = input("Seleccione una opción: ")

        if opcion == "1":
            print("Clasificando sin PCA...")
            self.knn.clasificar(usar_pca=False)
        elif opcion == "2":
            print("Clasificando con PCA...")
            self.knn.clasificar(usar_pca=True)
        else:
            print("Opción inválida.")

    def clasificar_audio_con_variables(self):
        """
        Clasifica el audio candidato utilizando las variables seleccionadas manualmente
        y grafica los resultados en 3D.
        """
        print("\n--- Clasificación del Audio Candidato con Variables Seleccionadas ---")
        # Define las variables seleccionadas manualmente
        variables = [2, 3, 7]  # Ejemplo: Centroide, Ancho de Banda, MFCC_3
        self.knn.seleccionar_variables(variables)
        print("Clasificando utilizando las variables seleccionadas...")
        prediccion = self.knn.clasificar()

        # Llamada al gráfico 3D
        print("Generando gráfico 3D con las variables seleccionadas...")
        X_train, y_train, X_test = self.knn.cargar_datos()
        self.knn.graficar_3d(X_train, y_train, X_test)

        print(f"El audio candidato fue clasificado como: {prediccion}")

    def aumentar_datos_db(self):
        """
        Aplica Data Augmentation sobre los audios de la base de datos crudos.
        """
        print("Aplicando Data Augmentation a los audios de la base de datos...")
        self.preprocesador.aumentar_datos(self.carpeta_aumentados)
        print(f"Los datos aumentados han sido guardados en la carpeta: {self.carpeta_aumentados}")

    def grabar_audio_crudo(self):
        """
        Graba un nuevo audio crudo para una verdura seleccionada por el usuario.
        """
        verdura = input("Ingrese el nombre de la verdura para la grabación (zanahoria, papa, camote, berenjena): ").strip().lower()

        if verdura not in ["zanahoria", "papa", "camote", "berenjena"]:
            print("Verdura no válida. Intente nuevamente.")
            return

        # Contar audios existentes de esta verdura
        audios_existentes = [f for f in os.listdir(self.carpeta_crudos) if f.startswith(verdura)]
        numero = len(audios_existentes) + 1
        nombre_archivo = f"{verdura} {numero}.wav"
        ruta_archivo = os.path.join(self.carpeta_crudos, nombre_archivo)

        print(f"Grabando audio para '{verdura}'...")

        try:
            self.grabadora.grabar_audio(verdura)
            os.rename(os.path.join(self.carpeta_candidato, "audio_candidato.wav"), ruta_archivo)
            print(f"Audio crudo guardado como: {ruta_archivo}")
        except Exception as e:
            print(f"Error al guardar el audio crudo: {e}")


# Función main para iniciar el programa
def main():
    """
    Función principal que inicia el sistema de clasificación de audios.
    """
    clasificador = ClasificadorAudios()
    clasificador.mostrar_menu()

if __name__ == "__main__":
    main()
