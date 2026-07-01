import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np
import threading

CLASS_NAMES = ["Bus", "Car", "Truck"]  # đúng thứ tự alphabet như lúc train
MODEL_PATH = "models/best_model.keras"

model = None  # load lazy sau khi UI hiện ra


def load_model_async():
    global model
    status_label.config(text="Đang load model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    status_label.config(text="Sẵn sàng. Hãy chọn ảnh.")
    btn_chon_anh.config(state="normal")


def du_doan(img: Image.Image):
    img_resized = img.convert("RGB").resize((224, 224))
    img_array = tf.keras.utils.img_to_array(img_resized)
    img_array = tf.expand_dims(img_array, 0)
    img_array = tf.keras.applications.efficientnet_v2.preprocess_input(img_array)

    predictions = model.predict(img_array, verbose=0)[0]
    class_idx = int(np.argmax(predictions))
    return CLASS_NAMES[class_idx], float(predictions[class_idx]) * 100, predictions


def chon_anh():
    path = filedialog.askopenfilename(
        title="Chọn ảnh xe",
        filetypes=[("Ảnh", "*.jpg *.jpeg *.png *.bmp"), ("Tất cả file", "*.*")]
    )
    if not path:
        return

    try:
        img = Image.open(path)
    except Exception as e:
        messagebox.showerror("Lỗi", f"Không mở được ảnh:\n{e}")
        return

    # Hiển thị ảnh (resize để vừa khung, giữ tỉ lệ)
    display_img = img.copy()
    display_img.thumbnail((400, 400))
    tk_img = ImageTk.PhotoImage(display_img)
    image_label.config(image=tk_img)
    image_label.image = tk_img  # giữ tham chiếu tránh bị garbage collect

    status_label.config(text="Đang phân tích...")
    root.update_idletasks()

    label, confidence, predictions = du_doan(img)

    result_label.config(text=f"Kết quả: {label}  ({confidence:.1f}%)")
    detail_text = "\n".join(
        f"{CLASS_NAMES[i]:6}: {predictions[i]*100:5.1f}%" for i in range(len(CLASS_NAMES))
    )
    detail_label.config(text=detail_text)
    status_label.config(text=f"Ảnh: {os.path.basename(path)}")


# ── UI ────────────────────────────────────────────────
root = tk.Tk()
root.title("Phân loại xe: Car / Bus / Truck")
root.geometry("460x650")
root.resizable(False, False)

tk.Label(root, text="🚗 Phân loại xe", font=("Segoe UI", 18, "bold")).pack(pady=15)

btn_chon_anh = tk.Button(
    root, text="Chọn ảnh...", font=("Segoe UI", 12),
    command=chon_anh, state="disabled", width=20, height=1
)
btn_chon_anh.pack(pady=10)

image_frame = tk.Frame(root, width=400, height=400, bg="#e0e0e0")
image_frame.pack(pady=10)
image_label = tk.Label(image_frame, bg="#e0e0e0")
image_label.pack(expand=True)

result_label = tk.Label(root, text="", font=("Segoe UI", 14, "bold"), fg="#1a7a1a")
result_label.pack(pady=10)

detail_label = tk.Label(root, text="", font=("Consolas", 11), justify="left")
detail_label.pack(pady=5)

status_label = tk.Label(root, text="Đang khởi động...", font=("Segoe UI", 9), fg="gray")
status_label.pack(pady=15)

# Load model trong thread riêng để UI không bị đứng
threading.Thread(target=load_model_async, daemon=True).start()

root.mainloop()
