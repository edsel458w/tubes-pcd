import numpy as np
import os
import joblib

MODEL_PATH = "model_knn.pkl"

KELAS_NAMA = {
    0: "Matang",
    1: "Mentah",
    2: "Terlalu Matang"
}


def model_tersedia():
    return os.path.exists(MODEL_PATH)


def klasifikasi_knn(fitur_list):
    """
    fitur_list: [R, G, B, H, S, V, Area]
    Returns: nama kelas hasil klasifikasi KNN K=5
    """
    if not model_tersedia():
        raise FileNotFoundError(
            "Model belum ada. Jalankan Training > Jalankan Training terlebih dahulu."
        )

    data = joblib.load(MODEL_PATH)
    knn = data["knn"]
    scaler = data["scaler"]

    fitur = np.array(fitur_list).reshape(1, -1)
    fitur_scaled = scaler.transform(fitur)
    pred = knn.predict(fitur_scaled)
    return KELAS_NAMA[pred[0]]
