import cv2

# TAHAP 1: Konversi RGB ke Grayscale
# sesuai jurnal: "mengubah citra menjadi grayscale"
# opencv bacanya BGR jadi pake COLOR_BGR2GRAY
def rgb_ke_grayscale(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray

# TAHAP 2: Konversi Grayscale ke Citra Biner
# sesuai jurnal: "dikonversikan menjadi citra biner"
# pake threshold 127 sebagai nilai tengah 0-255
# piksel di atas 127 jadi putih, di bawah jadi hitam

def grayscale_ke_biner(gray):
    _, biner = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    return biner