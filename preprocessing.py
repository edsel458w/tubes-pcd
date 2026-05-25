import cv2


# =============================================
# Konversi RGB ke Grayscale
# sesuai jurnal: "mengubah citra menjadi grayscale"
# opencv bacanya BGR jadi pake COLOR_BGR2GRAY
# =============================================
def rgb_ke_grayscale(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray


# =============================================
# FUNGSI BANTU: konversi BGR ke RGB
# dipanggil dari main.py buat nampilin gambar di PyQt
# opencv defaultnya BGR, PyQt butuh RGB
# =============================================
def bgr_ke_rgb(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb
