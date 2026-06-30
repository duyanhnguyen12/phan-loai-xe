import sys
import os
from pathlib import Path
import cv2
from ultralytics import YOLO

# COCO class ids cho xe
VEHICLE_CLASSES = {2: "Car", 5: "Bus", 7: "Truck"}

# Màu bounding box theo class
COLORS = {
    "Car":   (0, 200, 0),    # xanh lá
    "Bus":   (0, 120, 255),  # cam
    "Truck": (200, 0, 200),  # tím
}


def predict(image_path: str, output_dir: str = "output_yolo", conf: float = 0.3):
    img_path = Path(image_path)
    if not img_path.exists():
        print(f"[LỖI] Không tìm thấy ảnh: {image_path}")
        return

    model = YOLO("yolov8n.pt")  # tự tải về lần đầu (~6 MB)

    results = model(str(img_path), conf=conf, verbose=False, device="cpu")[0]

    img = cv2.imread(str(img_path))
    detections = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id not in VEHICLE_CLASSES:
            continue

        label = VEHICLE_CLASSES[cls_id]
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        color = COLORS[label]

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {confidence:.0%}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        detections.append((label, confidence))

    Path(output_dir).mkdir(exist_ok=True)
    out_path = Path(output_dir) / img_path.name
    cv2.imwrite(str(out_path), img)

    print(f"\n=== KẾT QUẢ: {img_path.name} ===")
    if detections:
        for label, conf_val in detections:
            print(f"  {label:6} — {conf_val:.1%}")
        print(f"  Tổng: {len(detections)} xe phát hiện")
    else:
        print("  Không phát hiện xe nào (thử giảm conf)")
    print(f"  Ảnh lưu tại: {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Chạy mặc định trên 3 ảnh test
        for img in ["test_car.jpg", "test_bus.jpg", "test_truck.jpg"]:
            if os.path.exists(img):
                predict(img)
    else:
        for arg in sys.argv[1:]:
            predict(arg)
