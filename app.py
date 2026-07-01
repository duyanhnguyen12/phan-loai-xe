import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

st.set_page_config(page_title="Phân loại xe", page_icon="🚗", layout="centered")

CLASS_NAMES = ["Bus", "Car", "Truck"]  # đúng thứ tự alphabet như lúc train


@st.cache_resource
def load_model():
    return tf.keras.models.load_model("models/best_model.keras")


def du_doan(model, img: Image.Image):
    img = img.convert("RGB").resize((224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    img_array = tf.keras.applications.efficientnet_v2.preprocess_input(img_array)

    predictions = model.predict(img_array, verbose=0)[0]
    class_idx = int(np.argmax(predictions))
    return CLASS_NAMES[class_idx], float(predictions[class_idx]) * 100, predictions


st.title("🚗 Phân loại xe: Car / Bus / Truck")
st.write("Chọn một ảnh xe để nhận diện.")

model = load_model()

uploaded_file = st.file_uploader("Chọn ảnh (jpg, png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Ảnh đã chọn", use_container_width=True)

    with st.spinner("Đang phân tích..."):
        label, confidence, predictions = du_doan(model, img)

    st.success(f"### Kết quả: **{label}** ({confidence:.1f}%)")

    st.write("#### Chi tiết xác suất từng lớp:")
    for i, cls in enumerate(CLASS_NAMES):
        st.write(f"{cls}: {predictions[i]*100:.1f}%")
        st.progress(float(predictions[i]))
