# classification.py
# klasifikasi wajah pake deepface - emosi, gender, usia
# cnn yang dipake: VGG-Face (deepface)

from deepface import DeepFace


def klasifikasi_wajah(roi_bgr):
    """
    nerima satu roi wajah (hasil crop dari segmentasi)
    return dict berisi emosi, gender, usia
    return None kalo gagal
    """
    try:
        hasil = DeepFace.analyze(
            roi_bgr,
            actions=["emotion", "age", "gender"],
            enforce_detection=False,  # ga error walau wajah blur/kecil
            silent=True               # biar ga spam log di terminal
        )

        # deepface kadang return list kalo nemu lebih dari 1 wajah
        if isinstance(hasil, list):
            hasil = hasil[0]

        emosi  = hasil.get("dominant_emotion", "unknown")
        usia   = hasil.get("age", 0)
        gender = hasil.get("dominant_gender", "unknown")

        # ambil semua skor emosi juga buat ditampilin di bar
        skor_emosi = hasil.get("emotion", {})

        return {
            "emosi": emosi,
            "usia": usia,
            "gender": gender,
            "skor_emosi": skor_emosi
        }

    except Exception as e:
        print(f"[klasifikasi] error: {e}")
        return None