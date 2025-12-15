# Pengenalan Emosi Musik Multimodal (Multimodal Music Emotion Recognition)

Proyek ini bertujuan untuk membangun sistem pengenalan emosi musik yang memanfaatkan pendekatan **Multimodal Intermediate Fusion**. Sistem ini menggabungkan fitur dari tiga modalitas berbeda: **Audio**, **Lirik**, dan **MIDI** untuk mengklasifikasikan emosi musik secara lebih akurat.

## Anggota Kemlompok
Muhammad Faqih Abdul Khobier (121140110)

Rayhan Fatih Gunawan (122140134)

Shintya Ayu Wardani (122140138)

Hizba Jaisy Muhammad (122140148)

Jonathan Jethro (122140213)





## 🚀 Fitur Utama

- **Multimodal Fusion**: Menggabungkan representasi fitur dari Audio (spektogram/CNN features), Lirik (embedding teks), dan MIDI (fitur simbolik) menggunakan arsitektur _Robust Concatenation Fusion_.
- **Strategi Pelatihan Canggih**:
  - **Stratified K-Fold Cross-Validation**: Menggunakan 5-fold CV untuk evaluasi yang lebih valid.
  - **Mixup Augmentation & Label Smoothing**: Meningkatkan generalisasi model dan mencegah overfitting.
  - **Modality Dropout**: Melatih model agar tetap tangguh meskipun salah satu modalitas hilang atau memiliki noise.
  - **Cosine Annealing Warm Restarts**: Penjadwalan learning rate untuk konvergensi yang lebih baik.
- **Ensemble & Test Time Augmentation (TTA)**: Menggabungkan prediksi dari beberapa model (fold) dan menerapkan TTA untuk inferensi yang lebih robust.

## 📂 Struktur Direktori

Berikut adalah struktur direktori utama proyek ini:

- `intermediate_fusion_final.py`: Skrip utama untuk pelatihan model, evaluasi, dan inferensi akhir.
- `pytorch/`: Berisi skrip terkait model PyTorch dan ekstraksi fitur audio.
  - `extract_features.py`: Skrip untuk mengekstrak fitur audio menggunakan model pre-trained (PANNs CNN14).
  - `check_features.py`: Utilitas untuk memeriksa fitur yang telah diekstrak.
- `utils/`: Folder utilitas (misal: konfigurasi, fungsi bantu).
- `ready_for_fusion/`: Direktori output tempat menyimpan embedding yang siap difusikan dan model yang telah dilatih.
- `Extracted_Feature/`: Direktori untuk menyimpan fitur mentah atau fitur yang telah diproses dari tiap modalitas.
- `dataset/`: Menyimpan dataset musik (audio, lirik, midi) dan file batch split kategori.

## 🛠️ Instalasi

Pastikan Anda memiliki Python 3.x terinstal. Instal dependensi yang diperlukan dengan perintah berikut:

```bash
pip install -r requirements.txt
```

Dependensi utama meliputi:

- `torch` (PyTorch)
- `librosa` & `soundfile` (Pemrosesan Audio)
- `numpy`, `scikit-learn` (Komputasi & Evaluasi)
- `matplotlib`, `seaborn` (Visualisasi)
- `tqdm` (Progress bar)

> **Catatan**: Untuk ekstraksi fitur audio, pastikan Anda juga memiliki model pre-trained `Cnn14_mAP=0.431.pth` di direktori yang sesuai (default: root atau folder `../`).

## 💻 Cara Penggunaan

### 1. Ekstraksi Fitur (Opsional/Jika belum ada)

Jika Anda belum memiliki fitur audio yang diekstrak, Anda dapat menjalankannya menggunakan skrip di folder `pytorch`:

```bash
python pytorch/extract_features.py --audio_dir dataset/Audio --output_path dataset/audio_features.npy
```

### 2. Pelatihan dan Evaluasi

Jalankan skrip utama untuk memulai proses pelatihan K-Fold Cross-Validation dan inferensi ensemble:

```bash
python intermediate_fusion_final.py
```

Skrip ini akan secara otomatis:

1.  Memuat embedding fitur (audio, lirik, midi).
2.  Melakukan normalisasi fitur.
3.  Membagi data menggunakan Stratified K-Fold.
4.  Melatih model untuk setiap fold dengan strategi Mixup dan Scheduler.
5.  Menyimpan model terbaik dari setiap fold ke folder `ready_for_fusion/`.
6.  Melakukan evaluasi akhir pada data uji (test set) menggunakan Ensemble dari 5 model fold + TTA.
7.  Menghasilkan kurva pembelajaran (`learning_curve.png`) dan confusion matrix (`confusion_matrix.png`).

## 📊 Hasil dan Visualisasi

Hasil pelatihan seperti plot loss/akurasi dan confusion matrix akan disimpan secara otomatis di direktori `ready_for_fusion/` (atau direktori output yang dikonfigurasi).

## � Lisensi

[MIT License](LICENSE.MIT)
