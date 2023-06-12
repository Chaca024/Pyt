import numpy as np
from sklearn.neural_network import MLPClassifier

# Data penduduk contoh
data_penduduk = np.array([[25, 1, 5000000],
                         [30, 0, 6000000],
                         [35, 1, 7000000],
                         [40, 0, 8000000]])

# Label klasifikasi untuk data penduduk
label = np.array([0, 0, 1, 1])

# Inisialisasi model JST
model = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000)

# Melatih model dengan data penduduk dan label
model.fit(data_penduduk, label)

# Data penduduk baru yang akan diklasifikasikan
data_penduduk_baru = np.array([[28, 1, 5500000],
                              [32, 0, 6500000]])

# Melakukan prediksi klasifikasi untuk data penduduk baru
klasifikasi = model.predict(data_penduduk_baru)

# Menampilkan hasil prediksi klasifikasi
for i in range(len(data_penduduk_baru)):
    print("Data penduduk", i+1, ":", data_penduduk_baru[i])
    if klasifikasi[i] == 0:
        print("Status: Penduduk tidak berpenghasilan")
    else:
        print("Status: Penduduk berpenghasilan")
    print()