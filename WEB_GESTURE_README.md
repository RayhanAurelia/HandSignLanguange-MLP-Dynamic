# 🤟 ASL Recognition Web - Gesture Blink Edition

Website interaktif untuk mendeteksi American Sign Language dengan fitur **Blink to Save** dan **Wave to Delete**.

## ✨ Fitur Utama

### 1. 👁️ Blink to Save

- **Kedipkan mata** untuk menyimpan huruf yang terdeteksi
- Sistem akan otomatis menambahkan huruf ke saved text
- Cooldown 10 frame untuk mencegah deteksi ganda

### 2. 👋 Wave to Delete (Undo)

- **Lambaikan tangan** secara horizontal untuk menghapus huruf terakhir
- Threshold: 80 pixel horizontal movement
- Cooldown 15 frame untuk mencegah spam

### 3. 🎨 Tema Black & White

- Desain minimalis hitam putih
- Animasi smooth dan interaktif
- Hover effects pada semua elemen
- Responsive design

## 🚀 Cara Menjalankan

```bash
# Install dependencies jika belum
pip install flask

# Jalankan server
python app.py
```

Buka browser: **http://localhost:5000**

## 🎯 Cara Menggunakan

1. **Posisikan tangan** di depan kamera dengan jelas
2. **Buat gesture** dari daftar sign yang tersedia
3. **Kedipkan mata** untuk menyimpan huruf yang terdeteksi
4. **Lambaikan tangan** untuk menghapus huruf terakhir
5. Klik **Clear Text** untuk menghapus semua teks

## 📊 Status Indikator

- **Current Detection**: Menampilkan huruf yang sedang terdeteksi
- **Confidence**: Bar menunjukkan tingkat kepercayaan prediksi
- **Saved Text**: Menampilkan teks yang sudah disimpan
- **Available Signs**: Daftar semua huruf yang dapat dideteksi

## 🎨 Desain

- **Background**: Pure Black (#000)
- **Cards**: Dark Gray (#111) dengan border (#333)
- **Text**: White (#FFF) dengan subtle gray (#999)
- **Animations**: Smooth transitions dan hover effects
- **Interactive**: Semua elemen responsif terhadap mouse hover

## 🔧 Technical Details

### Backend (app.py)

- Flask web server dengan video streaming
- MediaPipe untuk deteksi tangan dan wajah
- TensorFlow untuk prediksi gesture
- Real-time blink detection (eye aspect ratio < 0.20)
- Wave detection (horizontal wrist movement > 80px)

### Frontend

- HTML5 dengan Jinja2 templating
- Pure CSS3 dengan animations
- Vanilla JavaScript untuk fetch updates
- Update interval: 300ms

### Model Support

- ✅ Static Model (MLP)
- ✅ Dynamic Model (LSTM)
- Auto-detection berdasarkan input shape

## 📝 API Endpoints

- `GET /` - Main page
- `GET /video_feed` - Video streaming
- `GET /prediction` - Get current prediction + saved text (JSON)
- `POST /clear_text` - Clear saved text

## 🎮 Keyboard Shortcuts

Tidak ada keyboard shortcuts - semua kontrol melalui gesture dan blink!

## 🐛 Troubleshooting

### Blink tidak terdeteksi

- Pastikan wajah terlihat jelas di kamera
- Coba kedip lebih lama/jelas
- Periksa pencahayaan

### Wave tidak berfungsi

- Gerakan harus horizontal > 80 pixel
- Tunggu cooldown selesai (15 frame)
- Pastikan tangan terdeteksi

### Prediksi tidak akurat

- Hold gesture lebih lama
- Confidence harus > 50% untuk disimpan
- Pastikan pencahayaan cukup

## 🌟 Features

- [x] Real-time video streaming
- [x] Blink detection untuk save
- [x] Wave detection untuk undo
- [x] Smooth predictions dengan history
- [x] Black & white minimalist theme
- [x] Interactive animations
- [x] Confidence threshold (50%)
- [x] Cooldown system
- [x] Clear text button

Enjoy! 🚀
