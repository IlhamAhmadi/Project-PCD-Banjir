import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os
import gdown

# Konfigurasi Model
# Masukkan ID file Google Drive masing-masing model di sini
ID_MODEL_STD = '1Emb0EPkg1Pp8S8jLaQSzklMjpVMA878I'
ID_MODEL_ATT = '1YmePceW6eyW_KVH4O0CNmn3_cwWBKVrd'

PATH_STD = 'model_standard.h5'
PATH_ATT = 'model_attention.h5'

@st.cache_resource
def load_all_models():
    # Download Model Standard jika belum ada
    if not os.path.exists(PATH_STD):
        url_std = f'https://drive.google.com/uc?id={ID_MODEL_STD}'
        gdown.download(url_std, PATH_STD, quiet=False)
    
    # Download Model Attention jika belum ada
    if not os.path.exists(PATH_ATT):
        url_att = f'https://drive.google.com/uc?id={ID_MODEL_ATT}'
        gdown.download(url_att, PATH_ATT, quiet=False)
    
    # Load kedua model
    model_std = tf.keras.models.load_model(PATH_STD, compile=False)
    model_att = tf.keras.models.load_model(PATH_ATT, compile=False)
    return model_std, model_att

# Tampilan Dashboard
st.set_page_config(page_title="Perbandingan Deteksi Banjir", layout="wide")
st.title("Sistem Perbandingan Deteksi Banjir")
st.write("Analisis Segmentasi Citra: Standard U-Net vs Attention U-Net")

try:
    m_std, m_att = load_all_models()
    st.sidebar.write("Status: Semua Model Siap")
except Exception as e:
    st.sidebar.write(f"Status: Error - {e}")

uploaded_file = st.sidebar.file_uploader("Pilih gambar banjir", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image.convert('RGB'))
    
    # Preprocessing
    img_input = cv2.resize(img_array, (256, 256))
    img_input = img_input / 255.0
    img_input = np.expand_dims(img_input, axis=0)
    
    # Prediksi Kedua Model
    pred_std = m_std.predict(img_input)[0]
    pred_att = m_att.predict(img_input)[0]
    
    mask_std = (pred_std > 0.5).astype(np.uint8)
    mask_att = (pred_att > 0.5).astype(np.uint8)
    
    # Fungsi untuk membuat overlay merah
    def buat_overlay(img_original, mask_pred):
        m_resized = cv2.resize(mask_pred, (img_original.shape[1], img_original.shape[0]))
        ovl = img_original.copy()
        ovl[m_resized == 1] = [255, 0, 0]
        return cv2.addWeighted(img_original, 0.6, ovl, 0.4, 0)

    res_std = buat_overlay(img_array, mask_std)
    res_att = buat_overlay(img_array, mask_att)

    # Tampilan Layout 3 Kolom
    col_img, col_std, col_att = st.columns(3)
    
    with col_img:
        st.write("Gambar Asli")
        st.image(img_array, use_column_width=True)
        
    with col_std:
        st.write("Standard U-Net")
        st.image(res_std, use_column_width=True)
        
    with col_att:
        st.write("Attention U-Net")
        st.image(res_att, use_column_width=True)

    st.write("Keterangan: Area berwarna merah menunjukkan wilayah yang terdeteksi banjir oleh masing-masing model.")
else:
    st.write("Silakan unggah gambar untuk membandingkan hasil deteksi kedua arsitektur.")
