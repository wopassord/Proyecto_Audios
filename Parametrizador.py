import os
import numpy as np
import pandas as pd
import librosa
import re

class Parametrizador:
    def __init__(self, carpeta_procesados: str, archivo_parametros: str, n_mfcc: int = 20):
        """
        Clase para extraer parámetros de audios preprocesados y guardarlos en un archivo CSV.

        Args:
            carpeta_procesados (str): Ruta a la carpeta con los audios preprocesados.
            archivo_parametros (str): Ruta del archivo CSV donde se guardarán los parámetros.
            n_mfcc (int): Número de coeficientes MFCC a extraer (default: 13).
        """
        self.carpeta_procesados = carpeta_procesados
        self.archivo_parametros = archivo_parametros
        self.n_mfcc = n_mfcc

    def extraer_parametros(self):
        """
        Extrae los parámetros de los audios preprocesados y los guarda en un archivo CSV,
        sobrescribiendo el archivo existente.
        """
        archivos = [f for f in os.listdir(self.carpeta_procesados) if f.endswith('.wav')]
        if not archivos:
            print("No se encontraron audios preprocesados en la carpeta.")
            return

        datos = []

        for archivo in archivos:
            ruta_audio = os.path.join(self.carpeta_procesados, archivo)
            try:
                # Cargar el audio
                audio, sr = librosa.load(ruta_audio, sr=None)

                # Extraer parámetros
                zcr = np.mean(librosa.feature.zero_crossing_rate(audio))
                rms = np.mean(librosa.feature.rms(y=audio))
                mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=self.n_mfcc), axis=1)
                centroide = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
                ancho_banda = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr))

                # Obtener etiqueta del archivo
                etiqueta = self.extraer_etiqueta(archivo)

                # Concatenar parámetros
                parametros = [zcr, rms, centroide, ancho_banda] + list(mfccs) + [etiqueta]
                datos.append(parametros)

            except Exception as e:
                print(f"Error al procesar {archivo}: {e}")

        # Crear DataFrame y guardar en CSV
        columnas = ['ZCR', 'RMS', 'Centroide', 'Ancho_Banda'] + [f'MFCC_{i+1}' for i in range(self.n_mfcc)] + ['Etiqueta']
        df = pd.DataFrame(datos, columns=columnas)
        df.to_csv(self.archivo_parametros, index=False)  # Sobrescribe el archivo CSV
        print(f"Parámetros guardados en {self.archivo_parametros}")

    def extraer_etiqueta(self, archivo: str):
        """
        Extrae la etiqueta del nombre del archivo basado en las palabras clave.

        Args:
            archivo (str): Nombre del archivo de audio.

        Returns:
            str: Etiqueta (zanahoria, papa, camote, berenjena) o 'desconocido' si no coincide.
        """
        # Expresiones regulares para buscar las etiquetas
        patrones = ['zanahoria', 'papa', 'camote', 'berenjena']
        for patron in patrones:
            if re.search(patron, archivo, re.IGNORECASE):
                return patron.lower()
        return 'desconocido'

    def parametrizar_candidato(self):
        """
        Extrae los parámetros del audio candidato preprocesado y los guarda en un archivo CSV,
        sobrescribiendo el archivo existente.
        """
        archivo_candidato = os.path.join(self.carpeta_procesados, "processed_audio_candidato.wav")

        if not os.path.exists(archivo_candidato):
            print("No se encontró el archivo processed_audio_candidato.wav para parametrizar.")
            return

        try:
            # Cargar el audio candidato
            audio, sr = librosa.load(archivo_candidato, sr=None)

            # Extraer parámetros
            zcr = np.mean(librosa.feature.zero_crossing_rate(audio))
            rms = np.mean(librosa.feature.rms(y=audio))
            mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=self.n_mfcc, fmin=20, fmax=8000), axis=1)
            centroide = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
            ancho_banda = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr))

            # Concatenar parámetros en una sola fila
            parametros = [zcr, rms, centroide, ancho_banda] + list(mfccs)
            columnas = ['ZCR', 'RMS', 'Centroide', 'Ancho_Banda'] + [f'MFCC_{i+1}' for i in range(self.n_mfcc)]

            # Crear DataFrame y guardar en CSV
            df = pd.DataFrame([parametros], columns=columnas)
            df.to_csv(self.archivo_parametros, index=False)  # Sobrescribe el archivo CSV
            print(f"Parámetros del audio candidato guardados en {self.archivo_parametros}")

        except Exception as e:
            print(f"Error al parametrizar el audio candidato: {e}")
