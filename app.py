import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Konfigurasi halaman
st.set_page_config(
    page_title="Aplikasi Prediksi Penjualan",
    page_icon="📈",
    layout="centered"
)

# 2. Memuat model yang telah dilatih
@st.cache_resource
def load_model_packet():
    try:
        # Membaca paket model yang berisi model dan informasi kolom fitur
        paket = joblib.load('model_prediksi_penjualan.pth')
        return paket
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None

paket = load_model_packet()

# 3. Struktur UI Aplikasi
st.title("📈 Aplikasi Prediksi Penjualan Produk")
st.write("Aplikasi ini digunakan untuk memprediksi jumlah penjualan bulanan berdasarkan kategori produk, toko, dan promosi.")

if paket is not None:
    model = paket['model']
    model_name = paket['model_name']
    feature_columns = paket['feature_columns']
    
    st.success(f"Model Aktif: **{model_name}**")
    
    # Menyiapkan input form untuk user
    st.subheader("Masukkan Parameter Prediksi")
    st.caption("Catatan: model dilatih dengan data penjualan tahun 2013 (Januari-Desember), sehingga hasil prediksi untuk tahun lain hanya perkiraan.")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            store_nbr = st.number_input("Nomor Toko (Store Number)", min_value=1, max_value=54, value=1, step=1)
            year = st.number_input("Tahun", min_value=2013, max_value=2030, value=2014, step=1)
            month = st.slider("Bulan", min_value=1, max_value=12, value=1)
            
        with col2:
            onpromotion = st.number_input("Jumlah Produk Promo (On Promotion)", min_value=0, value=0, step=1)
            
            # List kategori family yang ada di dataset
            all_families = [
                col.replace('family_', '') for col in feature_columns if col.startswith('family_')
            ]
            family_selected = st.selectbox("Kategori Produk (Family)", sorted(all_families))
            
        submitted = st.form_submit_button("Prediksi Penjualan")
        
    if submitted:
        # 4. Melakukan data processing input user agar sesuai dengan format training
        input_data = pd.DataFrame([{ 
            'store_nbr': store_nbr,
            'onpromotion': onpromotion,
            'year': year,
            'month': month
        }])
        
        # Inisialisasi kolom encoding dummy kategori produk dengan nilai 0
        for col in feature_columns:
            if col.startswith('family_'):
                input_data[col] = 0
                
        # Set nilai 1 pada kategori yang dipilih user
        target_family_col = f"family_{family_selected}"
        if target_family_col in feature_columns:
            input_data[target_family_col] = 1
            
        # Memastikan urutan kolom input data sama persis dengan model
        input_data = input_data[feature_columns]
        
        # Melakukan prediksi
        prediksi = model.predict(input_data)[0]
        
        # Output hasil prediksi ke UI
        st.subheader("Hasil Prediksi:")
        st.metric(label=f"Estimasi Total Penjualan {family_selected}", value=f"{max(0, round(prediksi)):,}")
else:
    st.warning("Silakan pastikan file 'model_prediksi_penjualan.pth' berada di direktori yang sama dengan aplikasi ini.")
