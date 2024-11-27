import pickle
import numpy as np

# Cargar los datos desde el archivo pickle
with open('DatosGuardados/datos_lpd_5.pkl', 'rb') as f:
    data_list = pickle.load(f)

# Umbral para binarización (por ejemplo, si la intensidad es mayor que 0, la nota está activada)
threshold = 0

# Binarizar los datos
for row_data in data_list:
    # Binarizar pianoroll_0_csc_data (puedes aplicar esto a otros campos si es necesario)
    row_data['pianoroll_0_csc_data'] = (row_data['pianoroll_0_csc_data'] > threshold).astype(int)
    row_data['pianoroll_1_csc_data'] = (row_data['pianoroll_1_csc_data'] > threshold).astype(int)
    row_data['pianoroll_2_csc_data'] = (row_data['pianoroll_2_csc_data'] > threshold).astype(int)
    row_data['pianoroll_3_csc_data'] = (row_data['pianoroll_3_csc_data'] > threshold).astype(int)
    row_data['pianoroll_4_csc_data'] = (row_data['pianoroll_4_csc_data'] > threshold).astype(int)

    # Si también necesitas binarizar otros campos, puedes aplicar la misma lógica:
    # row_data['pianoroll_1_csc_data'] = (row_data['pianoroll_1_csc_data'] > threshold).astype(int)

# Guardar los datos binarizados en un archivo .pkl
with open('DatosGuardados/datos_lpd_5_binarizados.pkl', 'wb') as f:
    pickle.dump(data_list, f)

print("Datos binarizados y guardados en 'datos_lpd_5_binarizados.pkl'")
