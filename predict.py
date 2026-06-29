import tensorflow as tf
import numpy as np
from PIL import Image
import sys
import os

# Tắt log TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load model
print("Dang load model...")
model = tf.keras.models.load_model("models/best_model.keras")

CLASS_NAMES = ['Bus', 'Car', 'Truck']

def du_doan(duong_dan_anh):
    img = Image.open(duong_dan_anh).convert("RGB").resize((224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    predictions = model.predict(img_array, verbose=0)
    class_idx = int(np.argmax(predictions[0]))
    confidence = float(predictions[0][class_idx]) * 100

    print("\n=== KET QUA ===")
    print(f"Anh     : {duong_dan_anh}")
    print(f"Ket qua : {CLASS_NAMES[class_idx]}")
    print(f"Do tin cay: {confidence:.1f}%")
    print(f"Chi tiet: Bus={predictions[0][0]*100:.1f}% | Car={predictions[0][1]*100:.1f}% | Truck={predictions[0][2]*100:.1f}%")
    print("===============")

if len(sys.argv) > 1:
    du_doan(sys.argv[1])
else:
    print("Dung lenh: python predict.py <ten_anh.jpg>")