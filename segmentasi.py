import cv2
import numpy as np


# =============================================
# SEGMENTASI OTSU THRESHOLDING
# sesuai jurnal: "segmentasi citra menggunakan thresholding otsu"
# otsu nyari nilai threshold optimal secara otomatis
# background putih, objek pisang lebih gelap
# makanya pake THRESH_BINARY_INV biar objek jadi putih
# =============================================
def segmentasi_otsu(img):
    # konversi ke grayscale dulu sebelum otsu
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # otsu thresholding
    # nilai 0 di parameter kedua karna threshold dicari otomatis sama otsu
    nilai_threshold, hasil_otsu = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    return hasil_otsu, nilai_threshold


# =============================================
# OPERASI MORFOLOGI
# sesuai jurnal: "operasi morfologi untuk menyempurnakan hasil segmentasi"
# closing: nutup lubang kecil dalam objek pisang
# opening: buang noise kecil di luar objek
# =============================================
def closing(biner):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(biner, cv2.MORPH_CLOSE, kernel)


def opening(biner):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(biner, cv2.MORPH_OPEN, kernel)


def morfologi(biner):
    return opening(closing(biner))


# =============================================
# FUNGSI BANTU: terapkan mask ke gambar asli
# background jadi hitam, objek pisang tetap berwarna
# =============================================
def terapkan_mask(img, mask):
    hasil = cv2.bitwise_and(img, img, mask=mask)
    return hasil
