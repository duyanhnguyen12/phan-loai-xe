import sys
import os
from pathlib import Path

import torch
import torchvision.transforms.functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import cv2
import numpy as np

CLASSES    = ["__background__", "Bus", "Car", "Truck"]
COLORS     = {"Bus": (0, 120, 255), "Car": (0, 200, 0), "Truck": (200, 0, 200)}
MODEL_PATH = "models/best_model_rcnn.pth"
DEVICE     = torch.device("cpu")
CONF       = 0.5   # chỉ hiển thị detection có confidence >= 50%


def load_model():
    import sys
sys.path.insert(0, ".")
from train_rcnn import build_model
    model = build_model(len(CLASSES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def predict(image_path: str, output_dir: str = "output_rcnn"):
    img_path = Path(image_path)
    if not img_path.exists():
        print(f"[LỖI] Không tìm thấy ảnh: {image_path}")
        return

    model = load_model()

    img_pil = Image.open(img_path).convert("RGB")
    img_tensor = F.to_tensor(img_pil).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img_tensor)[0]

    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    detections = []
    for box, label, score in zip(outputs["boxes"], outputs["labels"], outputs["scores"]):
        if score < CONF:
            continue
        cls_name = CLASSES[label.item()]
        if cls_name == "__background__":
            continue

        x1, y1, x2, y2 = map(int, box.tolist())
        color = COLORS.get(cls_name, (255, 255, 255))

        cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 2)
        text = f"{cls_name} {score:.0%}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img_cv, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img_cv, text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        detections.append((cls_name, float(score)))

    Path(output_dir).mkdir(exist_ok=True)
    out_path = Path(output_dir) / img_path.name
    cv2.imwrite(str(out_path), img_cv)

    print(f"\n=== KẾT QUẢ: {img_path.name} ===")
    if detections:
        for cls_name, score in detections:
            print(f"  {cls_name:6} — {score:.1%}")
        print(f"  Tổng: {len(detections)} xe phát hiện")
    else:
        print("  Không phát hiện xe nào (thử giảm CONF trong file)")
    print(f"  Ảnh lưu tại: {out_path}")


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        for arg in sys.argv[1:]:
            predict(arg)
    else:
        print("=== Faster R-CNN Vehicle Detector ===")
        print("Nhập đường dẫn ảnh (hoặc 'q' để thoát)\n")
        while True:
            path = input("Ảnh: ").strip().strip('"')
            if path.lower() == "q":
                break
            if path:
                predict(path)
