import sys
import tkinter as tk
from tkinter import filedialog, messagebox
import threading

import torch
import torchvision.transforms.functional as F
from PIL import Image, ImageDraw, ImageFont, ImageTk

from train_rcnn import build_model
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))



CLASSES    = ["__background__", "Bus", "Car", "Truck"]
COLORS     = {"Bus": "#0078FF", "Car": "#00C800", "Truck": "#C800C8"}
MODEL_PATH = SCRIPT_DIR / "models" / "best_model_rcnn.pth"
DEVICE     = torch.device("cpu")
CONF       = 0.5   # chỉ hiển thị detection có confidence >= 50%

model = None  # load lazy sau khi UI hiện ra


def load_model_async():
    global model
    try:
        m = build_model(len(CLASSES))
        m.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        m.to(DEVICE)
        m.eval()
        model = m
        root.after(0, lambda: [
            status_label.config(text="Sẵn sàng. Hãy chọn ảnh."),
            btn_chon_anh.config(state="normal"),
        ])
    except Exception as e:
        root.after(0, lambda: messagebox.showerror("Lỗi load model", str(e)))  # noqa: F821


def du_doan(img: Image.Image):
    img_tensor = F.to_tensor(img.convert("RGB")).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img_tensor)[0]

    draw_img = img.convert("RGB").copy()
    draw = ImageDraw.Draw(draw_img)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    detections = []
    for box, label, score in zip(outputs["boxes"], outputs["labels"], outputs["scores"]):
        if score < CONF:
            continue
        cls_name = CLASSES[label.item()]
        if cls_name == "__background__":
            continue

        x1, y1, x2, y2 = map(int, box.tolist())
        color = COLORS.get(cls_name, "#FFFFFF")

        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        text = f"{cls_name} {score:.0%}"
        tb = draw.textbbox((0, 0), text, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        draw.rectangle([x1, y1 - th - 8, x1 + tw + 8, y1], fill=color)
        draw.text((x1 + 4, y1 - th - 6), text, fill="white", font=font)

        detections.append((cls_name, float(score)))

    return draw_img, detections


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

    status_label.config(text="Đang phân tích...")
    root.update_idletasks()

    try:
        result_img, detections = du_doan(img)
    except Exception as e:
        messagebox.showerror("Lỗi", f"Lỗi khi dự đoán:\n{e}")
        status_label.config(text="Sẵn sàng. Hãy chọn ảnh.")
        return

    # Hiển thị ảnh có bounding box, resize để vừa khung
    display_img = result_img.copy()
    display_img.thumbnail((420, 420))
    tk_img = ImageTk.PhotoImage(display_img)
    image_label.config(image=tk_img)
    image_label.image = tk_img

    if detections:
        detail_text = "\n".join(f"{cls:6}: {score:.1%}" for cls, score in detections)
        result_label.config(text=f"Phát hiện {len(detections)} xe", fg="#1a7a1a")
    else:
        detail_text = "Không phát hiện xe nào (confidence < 50%)"
        result_label.config(text="Không phát hiện", fg="#a83232")

    detail_label.config(text=detail_text)
    status_label.config(text=f"Ảnh: {Path(path).name}")


# ── UI ────────────────────────────────────────────────
root = tk.Tk()
root.title("Faster R-CNN - Phát hiện xe: Car / Bus / Truck")
root.geometry("480x680")
root.resizable(False, False)

tk.Label(root, text="🚗 Faster R-CNN Vehicle Detector", font=("Segoe UI", 16, "bold")).pack(pady=15)

btn_chon_anh = tk.Button(
    root, text="Chọn ảnh...", font=("Segoe UI", 12),
    command=chon_anh, state="disabled", width=20, height=1
)
btn_chon_anh.pack(pady=10)

image_frame = tk.Frame(root, width=420, height=420, bg="#e0e0e0")
image_frame.pack(pady=10)
image_frame.pack_propagate(False)
image_label = tk.Label(image_frame, bg="#e0e0e0")
image_label.pack(expand=True)

result_label = tk.Label(root, text="", font=("Segoe UI", 14, "bold"))
result_label.pack(pady=10)

detail_label = tk.Label(root, text="", font=("Consolas", 12), justify="left")
detail_label.pack(pady=5)

status_label = tk.Label(root, text="Đang khởi động...", font=("Segoe UI", 9), fg="gray")
status_label.pack(pady=15)

threading.Thread(target=load_model_async, daemon=True).start()

root.mainloop()
