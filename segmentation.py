# segmentation.py
# modul buat deteksi wajah pake yunet
# referensi: opencv zoo

import cv2
import numpy as np
import os
import urllib.request

MODEL_PATH = "face_detection_yunet_2023mar.onnx"
MODEL_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"

def download_model_kalo_belum_ada():
    if not os.path.exists(MODEL_PATH):
        print("downloading model dulu...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("done")

def init_detektor():
    download_model_kalo_belum_ada()
    det = cv2.FaceDetectorYN.create(
        MODEL_PATH,
        "",
        (640, 640),  # input size default, nanti diupdate
        score_threshold=0.5,
        nms_threshold=0.3,
        top_k=5000
    )
    return det

def deteksi_wajah(img_bgr, detektor):
    h, w = img_bgr.shape[:2]
    detektor.setInputSize((w, h))

    _, faces = detektor.detect(img_bgr)

    hasil = img_bgr.copy()
    rois = []
    boxes = []

    if faces is None:
        return hasil, rois, 0, boxes

    for i, face in enumerate(faces):
        x, y, fw, fh = int(face[0]), int(face[1]), int(face[2]), int(face[3])
        conf = face[14]

        # skip kalo confidence nya kecil
        if conf < 0.5:
            continue

        # benerin koordinat supaya ga out of bound
        x1 = max(x, 0)
        y1 = max(y, 0)
        x2 = min(x + fw, w)
        y2 = min(y + fh, h)

        # gambar kotak di wajah
        cv2.rectangle(hasil, (x1, y1), (x2, y2), (0, 200, 100), 2)

        # tulis label
        teks = f"wajah {i+1} ({conf:.2f})"
        cv2.putText(hasil, teks, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 100), 1)

        # crop bagian wajahnya
        roi = img_bgr[y1:y2, x1:x2]
        if roi.size > 0:
            rois.append(roi)
            boxes.append((x1, y1, x2 - x1, y2 - y1))

    jumlah = len(rois)

    # tampilin jumlah wajah di gambar
    cv2.putText(hasil, f"terdeteksi: {jumlah} wajah", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 220, 0), 2)

    return hasil, rois, jumlah, boxes