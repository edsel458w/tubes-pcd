import sys
import os
import cv2
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QFileDialog,
    QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QAction, QMessageBox
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

import imageacquisition as akuisisi
import preprocessing as prep
import segmentasi as seg
import feature_extraction as fitur
# import classification as klasifikasi  # TODO: buat classification.py


class AplikasiPisang(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Klasifikasi Kematangan Buah Pisang Ambon")
        self.resize(900, 650)

        self.img_asli = None
        self.mask = None
        self.hasil_segmentasi = None
        self.ciri_sekarang = None

        self.buat_menu()
        self.buat_tampilan()

        # self.cek_model_tersedia()  # TODO: butuh classification.py

    def buat_menu(self):
        menubar = self.menuBar()

        # ============ menu File ============
        menu_file = menubar.addMenu("File")

        # TODO: load_citra butuh imageacquisition.py (baca_gambar)
        aksi_load = QAction("Load Citra", self)
        aksi_load.setShortcut("Ctrl+O")
        aksi_load.triggered.connect(self.load_citra)
        menu_file.addAction(aksi_load)

        aksi_save = QAction("Save Hasil Segmentasi", self)
        aksi_save.setShortcut("Ctrl+S")
        aksi_save.triggered.connect(self.save_hasil)
        menu_file.addAction(aksi_save)

        menu_file.addSeparator()

        aksi_reset = QAction("Reset", self)
        aksi_reset.setShortcut("Ctrl+R")
        aksi_reset.triggered.connect(self.reset)
        menu_file.addAction(aksi_reset)

        menu_file.addSeparator()

        aksi_keluar = QAction("Keluar", self)
        aksi_keluar.setShortcut("Ctrl+Q")
        aksi_keluar.triggered.connect(self.close)
        menu_file.addAction(aksi_keluar)

        # ============ menu Training ============  # TODO: buat training.py & classification.py
        # menu_training = menubar.addMenu("Training")

        # aksi_training = QAction("Jalankan Training", self)
        # aksi_training.setShortcut("Ctrl+T")
        # aksi_training.triggered.connect(self.jalankan_training)
        # menu_training.addAction(aksi_training)

        # aksi_cek_model = QAction("Cek Status Model", self)
        # aksi_cek_model.triggered.connect(self.cek_status_model)
        # menu_training.addAction(aksi_cek_model)

        # ============ menu Preprocessing ============
        menu_prep = menubar.addMenu("Preprocessing")

        aksi_grayscale = QAction("Konversi Grayscale", self)
        aksi_grayscale.triggered.connect(self.proses_grayscale)
        menu_prep.addAction(aksi_grayscale)

        aksi_biner = QAction("Konversi Citra Biner", self)
        aksi_biner.triggered.connect(self.proses_biner)
        menu_prep.addAction(aksi_biner)

        # ============ menu Segmentasi ============
        menu_seg = menubar.addMenu("Segmentasi")

        aksi_otsu = QAction("Segmentasi Otsu", self)
        aksi_otsu.triggered.connect(self.proses_segmentasi)
        menu_seg.addAction(aksi_otsu)

        menu_morfologi = menu_seg.addMenu("Operasi Morfologi")

        aksi_closing = QAction("Closing", self)
        aksi_closing.triggered.connect(self.proses_closing)
        menu_morfologi.addAction(aksi_closing)

        aksi_opening = QAction("Opening", self)
        aksi_opening.triggered.connect(self.proses_opening)
        menu_morfologi.addAction(aksi_opening)

        # ============ menu Ekstraksi Ciri ============
        menu_ekstraksi = menubar.addMenu("Ekstraksi Ciri")

        aksi_ekstraksi = QAction("Ekstraksi RGB, HSV, Area", self)
        aksi_ekstraksi.triggered.connect(self.proses_ekstraksi)
        menu_ekstraksi.addAction(aksi_ekstraksi)

        # ============ menu Klasifikasi ============  # TODO: buat classification.py
        # menu_klasifikasi = menubar.addMenu("Klasifikasi")

        # aksi_knn = QAction("Klasifikasi KNN", self)
        # aksi_knn.triggered.connect(self.proses_klasifikasi)
        # menu_klasifikasi.addAction(aksi_knn)

        # ============ menu Bantuan ============
        menu_bantuan = menubar.addMenu("Bantuan")

        aksi_tentang = QAction("Tentang", self)
        aksi_tentang.triggered.connect(self.tentang)
        menu_bantuan.addAction(aksi_tentang)

    def buat_tampilan(self):
        widget_tengah = QWidget()
        self.setCentralWidget(widget_tengah)
        layout_utama = QVBoxLayout(widget_tengah)

        # ---- status nama file ----
        self.label_nama = QLabel("Belum ada gambar yang dimuat")
        self.label_nama.setAlignment(Qt.AlignCenter)
        layout_utama.addWidget(self.label_nama)

        # ---- area gambar ----
        baris_gambar = QHBoxLayout()

        kolom1 = QVBoxLayout()
        judul1 = QLabel("Citra Asli")
        judul1.setAlignment(Qt.AlignCenter)
        self.label_asli = QLabel()
        self.label_asli.setAlignment(Qt.AlignCenter)
        self.label_asli.setFixedSize(300, 300)
        self.label_asli.setStyleSheet("border: 1px solid gray")
        kolom1.addWidget(judul1)
        kolom1.addWidget(self.label_asli)

        kolom2 = QVBoxLayout()
        self.judul_hasil = QLabel("Hasil Proses")
        self.judul_hasil.setAlignment(Qt.AlignCenter)
        self.label_hasil_proses = QLabel()
        self.label_hasil_proses.setAlignment(Qt.AlignCenter)
        self.label_hasil_proses.setFixedSize(300, 300)
        self.label_hasil_proses.setStyleSheet("border: 1px solid gray")
        kolom2.addWidget(self.judul_hasil)
        kolom2.addWidget(self.label_hasil_proses)

        baris_gambar.addLayout(kolom1)
        baris_gambar.addLayout(kolom2)
        layout_utama.addLayout(baris_gambar)

        # ---- info proses ----
        self.label_info = QLabel("")
        self.label_info.setAlignment(Qt.AlignCenter)
        self.label_info.setStyleSheet("color: gray; font-size: 11px")
        layout_utama.addWidget(self.label_info)

        # ---- tabel ciri ----
        self.tabel = QTableWidget()
        self.tabel.setColumnCount(2)
        self.tabel.setHorizontalHeaderLabels(["Ciri", "Nilai"])
        self.tabel.setRowCount(7)
        nama_ciri = ["Red", "Green", "Blue", "Hue", "Saturation", "Value", "Area"]
        for i in range(7):
            self.tabel.setItem(i, 0, QTableWidgetItem(nama_ciri[i]))
        layout_utama.addWidget(self.tabel)

        # ---- hasil klasifikasi ----
        self.label_hasil = QLabel("Hasil Klasifikasi : -")
        self.label_hasil.setAlignment(Qt.AlignCenter)
        self.label_hasil.setStyleSheet("font-size: 18px; font-weight: bold")
        layout_utama.addWidget(self.label_hasil)

    # ====================== CEK MODEL ======================  # TODO: butuh classification.py
    # def cek_model_tersedia(self):
    #     if klasifikasi.model_tersedia():
    #         self.label_info.setText("Model KNN sudah tersedia dan siap digunakan")
    #     else:
    #         self.label_info.setText("Model belum ada - jalankan Training > Jalankan Training terlebih dahulu")

    # def cek_status_model(self):
    #     if klasifikasi.model_tersedia():
    #         QMessageBox.information(self, "Status Model", "Model KNN sudah tersedia\nFile: model_knn.pkl\n\nAplikasi siap melakukan klasifikasi")
    #     else:
    #         QMessageBox.warning(self, "Status Model", "Model KNN belum ada\n\nSilakan jalankan Training > Jalankan Training terlebih dahulu\nPastikan folder dataset/train sudah berisi gambar")

    # ====================== FUNGSI TRAINING ======================  # TODO: buat training.py & classification.py
    # def jalankan_training(self):
    #     folder_cek = [
    #         os.path.join("dataset", "train", "matang"),
    #         os.path.join("dataset", "train", "mentah"),
    #         os.path.join("dataset", "train", "terlalu_matang")
    #     ]
    #
    #     folder_kosong = []
    #     for f in folder_cek:
    #         if not os.path.exists(f):
    #             folder_kosong.append(f)
    #
    #     if len(folder_kosong) > 0:
    #         pesan = "Folder dataset tidak ditemukan:\n"
    #         for f in folder_kosong:
    #             pesan = pesan + "- " + f + "\n"
    #         pesan = pesan + "\nBuat folder tersebut dan isi dengan gambar pisang"
    #         QMessageBox.warning(self, "Folder Tidak Ditemukan", pesan)
    #         return
    #
    #     self.label_info.setText("Sedang training, harap tunggu...")
    #     QApplication.processEvents()
    #
    #     try:
    #         import training
    #         import importlib
    #         importlib.reload(training)
    #         self.label_info.setText("Training selesai! Model disimpan ke model_knn.pkl")
    #         QMessageBox.information(self, "Training Selesai", "Model KNN berhasil ditraining dan disimpan\nFile: model_knn.pkl")
    #     except Exception as e:
    #         self.label_info.setText("Training gagal: " + str(e))
    #         QMessageBox.warning(self, "Training Gagal", "Error: " + str(e))

    # ====================== FUNGSI MENU FILE ======================
    def load_citra(self):
        dialog = QFileDialog.getOpenFileName(self, "Pilih Gambar", "", "Images (*.png *.jpg *.jpeg)")
        file_path = dialog[0]
        if file_path == "":
            return

        img = akuisisi.baca_gambar(file_path)
        if img is None:
            self.label_nama.setText("Gagal membaca gambar")
            return

        self.img_asli = img
        nama = file_path.split("/")[-1]
        self.label_nama.setText("Gambar: " + nama)
        self.label_info.setText("")
        self.judul_hasil.setText("Hasil Proses")
        self.tampilkan_gambar(img, self.label_asli)

    def save_hasil(self):
        if self.hasil_segmentasi is None:
            QMessageBox.warning(self, "Peringatan", "Belum ada hasil yang bisa disimpan")
            return

        dialog = QFileDialog.getSaveFileName(self, "Simpan Gambar", "hasil.png", "Images (*.png *.jpg)")
        path_simpan = dialog[0]
        if path_simpan == "":
            return

        cv2.imwrite(path_simpan, self.hasil_segmentasi)
        QMessageBox.information(self, "Info", "Gambar berhasil disimpan")

    # ====================== FUNGSI PREPROCESSING ======================
    def proses_grayscale(self):
        if self.img_asli is None:
            QMessageBox.warning(self, "Peringatan", "Load citra terlebih dahulu melalui menu File > Load Citra")
            return

        gray = prep.rgb_ke_grayscale(self.img_asli)
        self.judul_hasil.setText("Hasil Grayscale")
        self.label_info.setText("Citra RGB dikonversi ke Grayscale (1 channel intensitas)")
        self.tampilkan_gambar(gray, self.label_hasil_proses)

    def proses_biner(self):
        if self.img_asli is None:
            QMessageBox.warning(self, "Peringatan", "Load citra terlebih dahulu melalui menu File > Load Citra")
            return

        gray = prep.rgb_ke_grayscale(self.img_asli)
        biner = prep.grayscale_ke_biner(gray)
        self.judul_hasil.setText("Hasil Citra Biner")
        self.label_info.setText("Grayscale dikonversi ke Biner dengan threshold = 127")
        self.tampilkan_gambar(biner, self.label_hasil_proses)

    # ====================== FUNGSI SEGMENTASI ======================
    def proses_segmentasi(self):
        if self.img_asli is None:
            QMessageBox.warning(self, "Peringatan", "Load citra terlebih dahulu melalui menu File > Load Citra")
            return

        hasil_otsu, nilai_t = seg.segmentasi_otsu(self.img_asli)
        self.mask = hasil_otsu

        self.judul_hasil.setText("Hasil Segmentasi Otsu")
        self.label_info.setText("Threshold Otsu optimal = " + str(round(nilai_t, 2)) + " (dicari otomatis)")
        self.tampilkan_gambar(hasil_otsu, self.label_hasil_proses)

    def proses_closing(self):
        if self.mask is None:
            QMessageBox.warning(self, "Peringatan", "Lakukan Segmentasi Otsu terlebih dahulu")
            return

        mask_closing = seg.closing(self.mask)
        self.mask = mask_closing

        hasil = seg.terapkan_mask(self.img_asli, mask_closing)
        self.hasil_segmentasi = hasil

        self.judul_hasil.setText("Hasil Closing")
        self.label_info.setText("Closing: menutup lubang kecil di dalam objek pisang")
        self.tampilkan_gambar(hasil, self.label_hasil_proses)

    def proses_opening(self):
        if self.mask is None:
            QMessageBox.warning(self, "Peringatan", "Lakukan Segmentasi Otsu terlebih dahulu")
            return

        mask_opening = seg.opening(self.mask)
        self.mask = mask_opening

        hasil = seg.terapkan_mask(self.img_asli, mask_opening)
        self.hasil_segmentasi = hasil

        self.judul_hasil.setText("Hasil Opening")
        self.label_info.setText("Opening: membuang noise kecil di luar objek pisang")
        self.tampilkan_gambar(hasil, self.label_hasil_proses)

    # ====================== FUNGSI EKSTRAKSI CIRI ======================
    def proses_ekstraksi(self):
        if self.mask is None:
            QMessageBox.warning(self, "Peringatan", "Lakukan segmentasi terlebih dahulu melalui menu Segmentasi")
            return

        ciri = fitur.ekstraksi_ciri(self.img_asli, self.mask)
        self.ciri_sekarang = ciri

        urutan = ["R", "G", "B", "H", "S", "V", "Area"]
        for i in range(len(urutan)):
            key = urutan[i]
            nilai = ciri[key]
            teks = str(nilai) if key == "Area" else str(round(nilai, 4))
            self.tabel.setItem(i, 1, QTableWidgetItem(teks))

        self.label_info.setText("Ekstraksi ciri RGB, HSV, dan Area dari area objek selesai")

    # ====================== FUNGSI KLASIFIKASI ======================  # TODO: buat feature_extraction.py & classification.py
    # def proses_klasifikasi(self):
    #     if self.ciri_sekarang is None:
    #         QMessageBox.warning(self, "Peringatan", "Lakukan ekstraksi ciri terlebih dahulu")
    #         return
    #
    #     if not klasifikasi.model_tersedia():
    #         QMessageBox.warning(self, "Model Belum Ada", "Jalankan Training > Jalankan Training terlebih dahulu")
    #         return
    #
    #     data_uji = fitur.ciri_ke_list(self.ciri_sekarang)
    #     hasil = klasifikasi.klasifikasi_knn(data_uji)
    #     self.label_hasil.setText("Hasil Klasifikasi : " + hasil)
    #     self.label_info.setText("KNN K=5 selesai | model dari model_knn.pkl")

    # ====================== FUNGSI BANTUAN ======================
    def tentang(self):
        teks = ("Aplikasi Klasifikasi Kematangan Buah Pisang Ambon\n"
                "Metode: K-Nearest Neighbor (KNN)\n"
                "Ciri: RGB, HSV, dan Area\n\n"
                "Alur penggunaan:\n"
                "1. Training > Jalankan Training (sekali saja)\n"
                "2. File > Load Citra\n"
                "3. Preprocessing > Grayscale > Biner\n"
                "4. Segmentasi > Otsu > Morfologi\n"
                "5. Ekstraksi Ciri\n"
                "6. Klasifikasi KNN")
        QMessageBox.information(self, "Tentang", teks)

    # ====================== RESET ======================
    def reset(self):
        self.img_asli = None
        self.mask = None
        self.hasil_segmentasi = None
        self.ciri_sekarang = None
        self.label_asli.clear()
        self.label_hasil_proses.clear()
        self.label_nama.setText("Belum ada gambar yang dimuat")
        self.label_hasil.setText("Hasil Klasifikasi : -")
        self.judul_hasil.setText("Hasil Proses")
        # self.cek_model_tersedia()  # TODO: butuh classification.py
        for i in range(7):
            self.tabel.setItem(i, 1, QTableWidgetItem(""))

    # ====================== FUNGSI BANTU TAMPIL GAMBAR ======================
    def tampilkan_gambar(self, img, label):
        if len(img.shape) == 2:
            h, w = img.shape
            qt_image = QImage(img.data, w, h, w, QImage.Format_Grayscale8)
        else:
            img_rgb = prep.bgr_ke_rgb(img)
            h, w, ch = img_rgb.shape
            bytes_per_line = ch * w
            qt_image = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(qt_image)
        pixmap = pixmap.scaled(300, 300, Qt.KeepAspectRatio)
        label.setPixmap(pixmap)


app = QApplication(sys.argv)
window = AplikasiPisang()
window.show()
sys.exit(app.exec_())
