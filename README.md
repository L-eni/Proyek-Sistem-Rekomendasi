# Proyek-Sistem-Rekomendasi

Proyek ini bertujuan untuk merancang dan mengembangkan model pembelajaran mesin yang dapat memberikan rekomendasi film kepada pengguna berdasarkan dataset yang tersedia.

## Struktur Proyek

- **Sistem_Rekomendasi_Film.ipynb** → Notebook yang memuat eksplorasi data, persiapan dataset, serta eksperimen dengan masing-masing model.
- **Laporan_sistem_Rekomendasi.md** → Laporan dokumentasi yang berisi analisis, kesimpulan, dan hasil dari proyek.
- **movies.csv**, **ratings.csv** → Dataset yang digunakan dalam proyek ini
- **requirements.txt** → Daftar paket dan dependensi yang perlu diinstal untuk menjalankan proyek.
- **Gambar/** → Folder yang menyimpan gambar-gambar yang digunakan dalam laporan untuk mendukung visualisasi analisis.

## Tujuan Proyek

- Menggunakan pendekatan **Collaborative Filtering** dengan membandingkan dua arsitektur model untuk memprediksi rating film.
- Membangun model **Matrix Factorization** (`RecommenderNet`) sebagai _baseline_, yang menggunakan _dot product_ antara _embedding_ pengguna dan film.
- Mengimplementasikan model **Neural Matrix Factorization (NeuMF)** yang menggabungkan jalur linear (GMF) dan non-linear (MLP) untuk menangkap pola preferensi yang lebih kompleks.
- Mengevaluasi kedua model menggunakan metrik **Root Mean Squared Error (RMSE)** dan **Mean Absolute Error (MAE)** untuk menentukan akurasi prediksi rating.
