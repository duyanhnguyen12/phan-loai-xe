import tensorflow as tf
import numpy as np
from PIL import Image
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load model
print("Dang load model...")
model = tf.keras.models.load_model("models/best_model.keras")
CLASS_NAMES = ['Bus', 'Car', 'Truck']

TEST_DIR = os.path.join("data", "test")

results = {'Bus': 0, 'Car': 0, 'Truck': 0}
files = [f for f in os.listdir(TEST_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))]

print(f"Tong so anh test: {len(files)}")
print("Dang du doan...\n")

for filename in files:
    path = os.path.join(TEST_DIR, filename)
    try:
        img = Image.open(path).convert("RGB").resize((224, 224))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

        predictions = model.predict(img_array, verbose=0)
        class_idx = int(np.argmax(predictions[0]))
        results[CLASS_NAMES[class_idx]] += 1
    except Exception as e:
        print(f"Loi anh {filename}: {e}")

# In kết quả
total = sum(results.values())
print("=" * 40)
print("KET QUA DU DOAN TOAN BO TAP TEST")
print("=" * 40)
for label, count in results.items():
    print(f"{label:10}: {count:4} anh ({count/total*100:.1f}%)")
print(f"{'Tong':10}: {total:4} anh")
print("=" * 40)