import pickle
from scipy.sparse import csc_matrix
import numpy as np
import matplotlib.pyplot as plt

# Configuración para imprimir matrices completas en consola
np.set_printoptions(threshold=np.inf)

# Cargar los datos del archivo .pkl
with open('DatosGuardados/datos_lpd_5.pkl', 'rb') as f:
    data_list = pickle.load(f)

# Extraer datos del primer elemento de data_list
data = data_list[0]
indptr = data['pianoroll_0_csc_indptr']
indices = data['pianoroll_0_csc_indices']
values = data['pianoroll_0_csc_data']
shape = data['pianoroll_0_csc_shape']

# Reconstruir la matriz completa
pianoroll_matrix = csc_matrix((values, indices, indptr), shape=shape).toarray()

# Imprimir la matriz completa
plt.figure(figsize=(10, 5))
plt.imshow(pianoroll_matrix, aspect='auto', cmap='gray_r')
plt.colorbar(label="Intensidad")
plt.title("Visualización de la matriz binarizada")
plt.xlabel("Columna (tiempo)")
plt.ylabel("Fila (notas)")
plt.show()




