import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Cargar los datos binarizados
with open('DatosGuardados/datos_lpd_5_binarizados.pkl', 'rb') as f:
    binarized_data_list = pickle.load(f)

# Parámetros de secuencia
sequence_length = 4  # Pasos de tiempo
note_range = (36, 84)  # Notas MIDI (C2 a C6)

# Listas para características y etiquetas
X = []
y = []

# Procesar cada archivo de datos binarizados
for row_data in binarized_data_list:
    # Reconstruir matriz
    shape = row_data['pianoroll_0_csc_shape']
    indices = row_data['pianoroll_0_csc_indices']
    data = row_data['pianoroll_0_csc_data']
    indptr = row_data['pianoroll_0_csc_indptr']

    pianoroll_matrix = np.zeros(shape, dtype=np.int8)
    for col in range(len(indptr) - 1):
        start_idx = indptr[col]
        end_idx = indptr[col + 1]
        pianoroll_matrix[indices[start_idx:end_idx], col] = data[start_idx:end_idx]

    # Reducir al rango deseado
    pianoroll_matrix = pianoroll_matrix[note_range[0]:note_range[1], :]

    # Generar secuencias
    for t in range(pianoroll_matrix.shape[1] - sequence_length):
        sequence = pianoroll_matrix[:, t:t + sequence_length]
        if sequence.shape == (note_range[1] - note_range[0], sequence_length):
            X.append(sequence.flatten())  # Aplanar secuencia
            next_note = pianoroll_matrix[:, t + sequence_length]
            y.append(np.argmax(next_note) if np.any(next_note) else -1)  # Nota más baja activa o -1 si no hay notas

# Filtrar casos con etiquetas inválidas (-1)
X, y = zip(*[(x, label) for x, label in zip(X, y) if label != -1])

# Convertir a NumPy
X = np.array(X, dtype=np.int8)
y = np.array(y, dtype=np.int8)

# Validar tamaños
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar modelo
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluar modelo
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
