# Aplikasi Prediksi Penjualan

Aplikasi web (Streamlit) untuk memprediksi total penjualan bulanan berdasarkan nomor toko, kategori produk (family), jumlah produk promo, tahun, dan bulan.

- **Model:** Random Forest Regressor (hasil Tugas Besar Machine Learning, dataset Store Sales)
- **File model:** `model_prediksi_penjualan.pth` (sudah termasuk di repository)

## Cara menjalankan
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Struktur
- `app.py` : kode aplikasi dan antarmuka
- `model_prediksi_penjualan.pth` : model terlatih + daftar fitur
- `requirements.txt` : dependensi
