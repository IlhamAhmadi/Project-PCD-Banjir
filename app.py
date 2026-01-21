import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os
import gdown

# Konfigurasi Model
# Masukkan ID file dari Google Drive di sini
MODEL_FILE_ID = '1Emb0EPkg1Pp8S8jLaQSzklMjpVMA878I' 
MODEL_PATH = 'model_banjir.h5'

@st.cache_resource
def load_trained_model():
    if not os.path.exists(MODEL_PATH):
        url = f'https://drive.google.com/uc?id={MODEL_FILE_ID}'
        gdown.download(url, MODEL_PATH, quiet=False)
    
    # Memuat model tanpa konfigurasi tambahan untuk stabilitas
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return model

# Tampilan Dashboard
st.set_page_config(page_title="Deteksi Banjir PCD", layout="wide")
st.title("Sistem Deteksi Banjir Otomatis")
st.write("Aplikasi Segmentasi Citra Menggunakan Arsitektur Attention U-Net")

try:
    model = load_trained_model()
    st.sidebar.write("Status: Model Berhasil Dimuat")
except Exception as e:
    st.sidebar.write(f"Status: Gagal Memuat Model - {e}")

# Menu Unggah Gambar
uploaded_file = st.sidebar.file_uploader("Pilih gambar banjir", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Konversi file ke array gambar
    image = Image.open(uploaded_file)
    img_array = np.array(image.convert('RGB'))
    
    # Preprocessing citra
    img_input = cv2.resize(img_array, (256, 256))
    img_input = img_input / 255.0
    img_input = np.expand_dims(img_input, axis=0)
    
    # Proses Prediksi
    prediction = model.predict(img_input)[0]
    mask = (prediction > 0.5).astype(np.uint8)
    
    # Penyesuaian ukuran mask ke ukuran asli gambar
    mask_resized = cv2.resize(mask, (img_array.shape[1], img_array.shape[0]))
    
    # Pembuatan Overlay
    overlay = img_array.copy()
    # Memberikan warna merah pada area banjir yang terdeteksi
    overlay[mask_resized == 1] = [255, 0, 0]
    
    # Penggabungan gambar asli dengan overlay transparan
    final_output = cv2.addWeighted(img_array, 0.6, overlay, 0.4, 0)

    # Menampilkan hasil pada dashboard
    col1, col2 = st.columns(2)
    with col1:
        st.write("Gambar Input")
        st.image(img_array, use_column_width=True)
    with col2:
        st.write("Hasil Deteksi Area Banjir")
        st.image(final_output, use_column_width=True)
else:
    st.write("Silakan unggah gambar melalui menu di samping untuk memulai analisis.")
