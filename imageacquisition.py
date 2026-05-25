import cv2

def load_image(path):
    img = cv2.imread(path)
    if img is None:
        print("Gagal membaca gambar")
    return img

def baca_gambar(path):
    img = cv2.imread(path)
    if img is None:
        print("Gagal membaca gambar: " + path)
    return img

