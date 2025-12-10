# American Sign Language Recognition - Web Application

Website untuk mendeteksi American Sign Language secara real-time menggunakan webcam.

## Cara Menjalankan Website

### 1. Install Dependencies

Pastikan semua library terinstall:

```bash
pip install -r requirements.txt
```

### 2. Jalankan Web Server

```bash
python app.py
```

### 3. Buka Browser

Buka browser dan akses:

```
http://localhost:5000
```

## Fitur Website

- **Real-time Video Streaming**: Menampilkan feed dari webcam dengan deteksi tangan
- **Prediksi Langsung**: Menampilkan prediksi sign language secara real-time
- **Confidence Score**: Menampilkan tingkat kepercayaan prediksi
- **Visual Hand Landmarks**: Menampilkan titik-titik landmark pada tangan
- **Responsive Design**: Tampilan yang menarik dan responsif

## Struktur File

```
hand-sign-real-time/
├── app.py                 # Flask web application
├── templates/
│   └── index.html        # Halaman utama website
├── static/
│   └── style.css         # Styling untuk website
├── model2/
│   ├── model.h5          # Model terlatih
│   ├── scaler.pkl        # Scaler untuk preprocessing
│   └── label_binarizer.pkl  # Label encoder
└── utils/
    ├── config.py         # Konfigurasi
    └── distance.py       # Fungsi ekstraksi fitur
```

## Cara Menggunakan

1. **Izinkan Akses Kamera**: Saat website dibuka, izinkan akses ke webcam
2. **Posisikan Tangan**: Letakkan tangan di depan kamera dengan jelas
3. **Buat Gesture**: Buat gesture dari daftar sign yang tersedia
4. **Lihat Hasil**: Prediksi akan muncul secara real-time dengan confidence score

## Teknologi yang Digunakan

- **Backend**: Flask (Python)
- **Deep Learning**: TensorFlow/Keras
- **Computer Vision**: OpenCV, MediaPipe
- **Frontend**: HTML, CSS, JavaScript
- **Preprocessing**: Scikit-learn

## Troubleshooting

### Kamera tidak terdeteksi

- Pastikan webcam terhubung dengan baik
- Coba ubah index kamera di `utils/config.py`

### Error saat load model

- Pastikan file model ada di folder `model2/`
- Pastikan file `scaler.pkl` dan `label_binarizer.pkl` ada

### Prediksi tidak akurat

- Pastikan pencahayaan cukup
- Tangan harus terlihat jelas di frame
- Hindari background yang kompleks

## Notes

Website ini menggunakan model yang telah dilatih sebelumnya. Pastikan model sudah dilatih terlebih dahulu menggunakan `train.py` sebelum menjalankan website.
