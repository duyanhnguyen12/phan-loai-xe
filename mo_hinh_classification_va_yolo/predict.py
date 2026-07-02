import tensorflow as tf
import numpy as np
from PIL import Image
import sys
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("Dang load model...")
model = tf.keras.models.load_model("models/best_model.keras")

temp_ds = tf.keras.utils.image_dataset_from_directory(
    "../data/train",
    image_size=(224, 224),
    batch_size=1,
    shuffle=False,
    verbose=False
)
CLASS_NAMES = temp_ds.class_names
print(f"[INFO] Thứ tự nhãn: {CLASS_NAMES}")

def du_doan(duong_dan_anh):
    img = Image.open(duong_dan_anh).convert("RGB").resize((224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    img_array = tf.keras.applications.efficientnet_v2.preprocess_input(img_array)

    predictions = model.predict(img_array, verbose=0)
    class_idx = int(np.argmax(predictions[0]))
    confidence = float(predictions[0][class_idx]) * 100

    print("\n=== KET QUA ===")
    print(f"Anh       : {duong_dan_anh}")
    print(f"Ket qua   : {CLASS_NAMES[class_idx]}")
    print(f"Do tin cay: {confidence:.1f}%")
    print(f"Chi tiet  : { ' | '.join(f'{CLASS_NAMES[i]}={predictions[0][i]*100:.1f}%' for i in range(len(CLASS_NAMES))) }")
    print("===============")

if len(sys.argv) > 1:
    du_doan(sys.argv[1])
else:
    print("Dung lenh: python predict.py <ten_anh.jpg>")
