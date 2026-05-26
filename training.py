import cv2
import numpy as np
import os
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

import segmentasi as seg
import feature_extraction as fitur_ext

DATASET_TRAIN = "dataset/train"
MODEL_PATH = "model_knn.pkl"
K = 5

KELAS_MAP = {
    "matang": 0,
    "mentah": 1,
    "terlalu-matang": 2
}
KELAS_NAMA = ["Matang", "Mentah", "Terlalu Matang"]


def _ekstrak_folder(folder):
    X, y = [], []
    for kelas in sorted(os.listdir(folder)):
        if kelas not in KELAS_MAP:
            continue
        kelas_folder = os.path.join(folder, kelas)
        if not os.path.isdir(kelas_folder):
            continue
        for file in sorted(os.listdir(kelas_folder)):
            path = os.path.join(kelas_folder, file)
            img = cv2.imread(path)
            if img is None:
                continue
            # Sesuai jurnal: segmentasi Otsu + closing sebelum ekstraksi ciri
            mask_otsu, _ = seg.segmentasi_otsu(img)
            mask = seg.closing(mask_otsu)
            ciri = fitur_ext.ekstraksi_ciri(img, mask)
            X.append(fitur_ext.ciri_ke_list(ciri))
            y.append(KELAS_MAP[kelas])
    return np.array(X), np.array(y)


def jalankan_training():
    if not os.path.exists(DATASET_TRAIN):
        raise FileNotFoundError(f"Folder {DATASET_TRAIN} tidak ditemukan")

    X_train, y_train = _ekstrak_folder(DATASET_TRAIN)
    if len(X_train) == 0:
        raise ValueError("Tidak ada data latih yang berhasil dibaca")

    # Normalisasi semua fitur ke rentang 0-1 (termasuk Area)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_train)

    # PCA 2 komponen untuk visualisasi sebaran data (sesuai jurnal)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Plot sebaran data latih
    plt.figure(figsize=(8, 6))
    plt.title("Sebaran Data Pelatihan KNN")
    warna_latih = ["red", "green", "blue"]
    for label, nama, warna in zip([0, 1, 2], KELAS_NAMA, warna_latih):
        idx = y_train == label
        plt.scatter(X_pca[idx, 0], X_pca[idx, 1], c=warna, label=nama)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.5)

    # KNN K=5 dengan Euclidean Distance (sesuai jurnal)
    knn = KNeighborsClassifier(n_neighbors=K, metric="euclidean")
    knn.fit(X_scaled, y_train)

    # Simpan model: KNN + scaler untuk normalisasi + PCA untuk visualisasi
    joblib.dump({"knn": knn, "scaler": scaler, "pca": pca}, MODEL_PATH)

    return {
        "jumlah_latih": len(X_train),
    }


if __name__ == "__main__":
    info = jalankan_training()
    print(f"Training selesai. Jumlah data latih: {info['jumlah_latih']}")
    input("Tekan Enter untuk keluar...")
