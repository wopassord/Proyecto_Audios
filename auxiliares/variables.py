import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def identificar_variables_diferenciadoras(csv_path, categoria1, categoria2):
    """
    Identifica las variables que más diferencian entre dos categorías.

    Args:
        csv_path (str): Ruta al archivo CSV con los parámetros de los audios.
        categoria1 (str): Nombre de la primera categoría.
        categoria2 (str): Nombre de la segunda categoría.

    Returns:
        pd.DataFrame: Tabla con las variables y sus diferencias promedio entre las categorías.
    """
    try:
        # Cargar datos
        df = pd.read_csv(csv_path)

        # Separar etiquetas y datos
        etiquetas = df.iloc[:, -1]  # Última columna contiene las etiquetas
        datos = df.iloc[:, :-1]    # Todas las columnas excepto la última son variables

        # Filtrar categorías de interés
        datos_categoria1 = datos[etiquetas == categoria1]
        datos_categoria2 = datos[etiquetas == categoria2]

        # Normalizar los datos en el rango [0, 1]
        scaler = MinMaxScaler()
        datos_normalizados = scaler.fit_transform(datos)

        # Reconstruir DataFrame normalizado
        df_normalizado = pd.DataFrame(datos_normalizados, columns=datos.columns)
        df_normalizado["Etiqueta"] = etiquetas.values

        # Recalcular datos normalizados por categoría
        datos_categoria1_norm = df_normalizado[df_normalizado["Etiqueta"] == categoria1].iloc[:, :-1]
        datos_categoria2_norm = df_normalizado[df_normalizado["Etiqueta"] == categoria2].iloc[:, :-1]

        # Calcular diferencias promedio por variable
        media_categoria1 = datos_categoria1_norm.mean()
        media_categoria2 = datos_categoria2_norm.mean()
        diferencias = abs(media_categoria1 - media_categoria2)

        # Crear un DataFrame de resultados
        resultados = pd.DataFrame({
            "Variable": datos.columns,
            "Diferencia Media": diferencias
        }).sort_values(by="Diferencia Media", ascending=False)

        return resultados

    except Exception as e:
        print(f"Error: {e}")
        return None


# Configuración inicial
csv_path = "./DB/parametros_DB.csv"  # Ruta al archivo CSV
categoria1 = "zanahoria"             # Primera categoría
categoria2 = "camote"                # Segunda categoría

# Llamar a la función y mostrar resultados
resultados = identificar_variables_diferenciadoras(csv_path, categoria1, categoria2)
if resultados is not None:
    print("\nVariables que más diferencian entre las categorías:")
    print(resultados)
