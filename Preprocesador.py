import os
import librosa
import soundfile as sf
import numpy as np

class Preprocesador:
    def __init__(self, carpeta_crudos: str, carpeta_procesados: str, frecuencia_muestreo: int = 16000):
        """
        Clase encargada de preprocesar audios crudos.

        Args:
            carpeta_crudos (str): Ruta a la carpeta con los audios crudos.
            carpeta_procesados (str): Ruta a la carpeta para guardar los audios preprocesados.
            frecuencia_muestreo (int): Frecuencia de muestreo deseada (default: 16000 Hz).
        """
        self.carpeta_crudos = carpeta_crudos
        self.carpeta_procesados = carpeta_procesados
        self.frecuencia_muestreo = frecuencia_muestreo

    def procesar_audios(self):
        """
        Procesa todos los audios en la carpeta crudos y los guarda en la carpeta procesados.
        """
        if not os.path.exists(self.carpeta_procesados):
            os.makedirs(self.carpeta_procesados)

        archivos = [f for f in os.listdir(self.carpeta_crudos) if f.endswith('.wav')]
        if not archivos:
            print("No se encontraron archivos .wav en la carpeta crudos.")
            return

        for archivo in archivos:
            ruta_crudo = os.path.join(self.carpeta_crudos, archivo)
            ruta_procesado = os.path.join(self.carpeta_procesados, archivo)

            try:
                # Cargar el archivo
                audio, sr = librosa.load(ruta_crudo, sr=self.frecuencia_muestreo, mono=True)
                
                # Normalizar la amplitud entre -1 y 1
                audio = librosa.util.normalize(audio)

                # Guardar el audio procesado
                sf.write(ruta_procesado, audio, self.frecuencia_muestreo)
                print(f"Audio procesado y guardado: {ruta_procesado}")

            except Exception as e:
                print(f"Error al procesar {archivo}: {e}")

    def aumentar_datos(self, carpeta_aumentados: str):
        """
        Realiza Data Augmentation sobre los audios en la carpeta crudos y guarda las nuevas versiones
        en una carpeta de datos aumentados.

        Args:
            carpeta_aumentados (str): Ruta a la carpeta para guardar los audios aumentados.
        """
        if not os.path.exists(carpeta_aumentados):
            os.makedirs(carpeta_aumentados)

        archivos = [f for f in os.listdir(self.carpeta_crudos) if f.endswith('.wav')]
        if not archivos:
            print("No se encontraron archivos .wav en la carpeta crudos.")
            return

        for archivo in archivos:
            ruta_crudo = os.path.join(self.carpeta_crudos, archivo)
            try:
                # Cargar el archivo
                audio, sr = librosa.load(ruta_crudo, sr=self.frecuencia_muestreo, mono=True)

                # Normalizar la amplitud
                audio = librosa.util.normalize(audio)

                # Generar datos aumentados
                aumentados = self._generar_aumentaciones(audio, sr)

                # Guardar los audios aumentados
                for i, audio_aug in enumerate(aumentados):
                    nombre_aumentado = archivo.replace(".wav", f"_aug{i+1}.wav")
                    ruta_aumentado = os.path.join(carpeta_aumentados, nombre_aumentado)
                    sf.write(ruta_aumentado, audio_aug, self.frecuencia_muestreo)
                    print(f"Audio aumentado guardado: {ruta_aumentado}")

            except Exception as e:
                print(f"Error al aumentar {archivo}: {e}")

    def _generar_aumentaciones(self, audio, sr):
        """
        Genera versiones aumentadas de un audio mediante diversas transformaciones.

        Args:
            audio (np.array): Señal de audio original.
            sr (int): Frecuencia de muestreo.

        Returns:
            list: Lista de señales de audio aumentadas.
        """
        aumentados = []

        # Agregar ruido blanco
        ruido = np.random.normal(0, 0.005, audio.shape)
        aumentados.append(audio + ruido)

        # Cambio de tono (Pitch Shifting)
        aumentados.append(librosa.effects.pitch_shift(audio, sr=sr, n_steps=2))
        aumentados.append(librosa.effects.pitch_shift(audio, sr=sr, n_steps=-2))

        # Estiramiento temporal (Time Stretching)
        aumentados.append(librosa.effects.time_stretch(audio, rate=1.1))
        aumentados.append(librosa.effects.time_stretch(audio, rate=0.9))

        return aumentados
