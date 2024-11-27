import os
import pickle

import numpy as np

# Directorio donde se encuentran los archivos .npz
folder_path = "C:/Users/usuario/Desktop/Analisis de datos/Analisis/lpd_5_cleansed"

# Lista para almacenar los datos de cada archivo
data_list = []

# Recorremos todas las carpetas dentro del directorio
for root, dirs, files in os.walk(folder_path):
    for file in files:
        if file.endswith(".npz"):  # Solo leer archivos .npz
            file_path = os.path.join(root, file)
            print(f"Cargando archivo: {file_path}")

            # Cargar el archivo npz
            data = np.load(file_path)

            # Extraer los componentes y agregarlos a una lista
            row_data = {}
            for key in data.keys():
                row_data[key] = data[key]

            # Agregar la fila a la lista de datos
            data_list.append(row_data)

# Guardar la lista de diccionarios como un archivo .pkl
with open('DatosGuardados/datos_lpd_5.pkl', 'wb') as f:
    pickle.dump(data_list, f)

print("Datos guardados en 'datos_lpd_5.pkl'")
