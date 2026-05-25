import cv2
import numpy as np


# ekstraksi ciri dari gambar yg udah disegmentasi
# yg diambil cuma bagian objek (pisang) bukan background hitam
# ngikutin jurnal: ciri R,G,B,H,S,V (dinormalisasi 0-1) + Area
def ekstraksi_ciri(img, mask):
    # ambil pixel yg masuk dalam mask aja (objek pisang)
    # mask > 0 artinya bagian objek
    area_objek = mask > 0

    # ---- ciri RGB ----
    # opencv format BGR jadi index 2=R, 1=G, 0=B
    b = img[:, :, 0]
    g = img[:, :, 1]
    r = img[:, :, 2]

    # rata2 nilai tiap channel di area objek aja
    # dibagi 255 biar jadi 0-1 kaya di jurnal
    rata_r = np.mean(r[area_objek]) / 255
    rata_g = np.mean(g[area_objek]) / 255
    rata_b = np.mean(b[area_objek]) / 255

    # ---- ciri HSV ----
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    # opencv: H rangenya 0-179, S sama V 0-255
    # makanya pembaginya beda biar sama2 jadi 0-1
    rata_h = np.mean(h[area_objek]) / 179
    rata_s = np.mean(s[area_objek]) / 255
    rata_v = np.mean(v[area_objek]) / 255

    # ---- ciri Area ----
    # area = jumlah pixel objek (yg masuk mask)
    area = int(np.sum(area_objek))

    # balikin dalam bentuk dictionary biar gampang dibaca di GUI
    ciri = {
        "R": rata_r,
        "G": rata_g,
        "B": rata_b,
        "H": rata_h,
        "S": rata_s,
        "V": rata_v,
        "Area": area
    }

    return ciri


# ubah dictionary ciri jadi list angka aja
# buat dipake di perhitungan KNN
def ciri_ke_list(ciri):
    data = [
        ciri["R"],
        ciri["G"],
        ciri["B"],
        ciri["H"],
        ciri["S"],
        ciri["V"],
        ciri["Area"]
    ]
    return data
