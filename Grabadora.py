import os
import sounddevice as sd
import librosa
import soundfile as sf

class Grabadora:
    def __init__(self, carpeta_candidato: str, frecuencia_muestreo: int = 16000, duracion: int = 3):
        """
        Clase para grabar audios y preprocesarlos.

        Args:
            carpeta_candidato (str): Ruta a la carpeta donde se guardarán los audios del candidato.
            frecuencia_muestreo (int): Frecuencia de muestreo deseada para los audios grabados (default: 16000 Hz).
            duracion (int): Duración en segundos de la grabación (default: 3 segundos).
        """
        self.carpeta_candidato = carpeta_candidato
        self.frecuencia_muestreo = frecuencia_muestreo
        self.duracion = duracion

    def grabar_audio(self, verdura: str):
        """
        Graba un audio y lo guarda como 'audio_candidato.wav'.

        Args:
            verdura (str): Nombre de la verdura seleccionada para asociar con el audio grabado.
        """
        if not os.path.exists(self.carpeta_candidato):
            os.makedirs(self.carpeta_candidato)

        ruta_audio = os.path.join(self.carpeta_candidato, "audio_candidato.wav")

        try:
            print(f"Grabando audio para '{verdura}' por {self.duracion} segundos...")
            audio = sd.rec(int(self.duracion * self.frecuencia_muestreo), samplerate=self.frecuencia_muestreo, channels=1, dtype='float32')
            sd.wait()
            sf.write(ruta_audio, audio, self.frecuencia_muestreo)
            print(f"Audio grabado y guardado en: {ruta_audio}")
        except Exception as e:
            print(f"Error al grabar el audio: {e}")

    def preprocesar_audio_candidato(self):
        """
        Preprocesa el audio grabado para garantizar un formato homogéneo.
        """
        ruta_audio_crudo = os.path.join(self.carpeta_candidato, "audio_candidato.wav")
        ruta_audio_procesado = os.path.join(self.carpeta_candidato, "processed_audio_candidato.wav")

        if not os.path.exists(ruta_audio_crudo):
            print("No se encontró el audio candidato para preprocesar.")
            return

        try:
            # Cargar y procesar el audio
            audio, sr = librosa.load(ruta_audio_crudo, sr=self.frecuencia_muestreo, mono=True)
            audio = librosa.util.normalize(audio)

            # Guardar el audio procesado
            sf.write(ruta_audio_procesado, audio, self.frecuencia_muestreo)
            print(f"Audio preprocesado y guardado en: {ruta_audio_procesado}")
        except Exception as e:
            print(f"Error al preprocesar el audio candidato: {e}")

# Función main para pruebas unitarias
def main():
    """
    Prueba unitaria de la clase Grabadora.
    - Permite grabar un audio para una verdura seleccionada.
    - Preprocesa el audio grabado.
    """
    carpeta_candidato = os.path.join(os.getcwd(), "Candidato")
    grabadora = Grabadora(carpeta_candidato)

    # Selección de verdura
    verdura = input("Ingrese el nombre de la verdura para la grabación (ejemplo: zanahoria): ").strip()
    grabadora.grabar_audio(verdura)
    grabadora.preprocesar_audio_candidato()

if __name__ == "__main__":
    main()
