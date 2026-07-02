import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image
from ultralytics import YOLO

# COCO class ids cho xe
VEHICLE_CLASSES = {2: "Car", 5: "Bus", 7: "Truck"}

# Thư mục cần annotate
DATA_DIRS = ["data/train", "data/val"]

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def annotate_dataset(conf: float = 0.3):
    model = YOLO("yolov8n.pt")

    total_annotated = 0
    total_skipped = 0

    for data_dir in DATA_DIRS:
        for cls_name in ["Bus", "Car", "Truck"]:
            cls_dir = Path(data_dir) / cls_name
            if not cls_dir.exists():
                continue

            images = [f for f in cls_dir.iterdir() if f.suffix.lower() in EXTENSIONS]
            print(f"\n[{data_dir}/{cls_name}] Đang annotate {len(images)} ảnh...")

            annotated = skipped = 0

            for img_path in images:
                xml_path = img_path.with_suffix(".xml")
                if xml_path.exists():
                    skipped += 1
                    continue

                # YOLO detect
                results = model(str(img_path), conf=conf, verbose=False, device="cpu")[0]

                # Lọc box xe, ưu tiên box có class khớp với thư mục
                boxes = []
                for box in results.boxes:
                    cid = int(box.cls[0])
                    if cid not in VEHICLE_CLASSES:
                        continue
                    boxes.append({
                        "label": VEHICLE_CLASSES[cid],
                        "conf": float(box.conf[0]),
                        "xyxy": list(map(int, box.xyxy[0].tolist()))
                    })

                if not boxes:
                    # Không detect được → dùng toàn bộ ảnh làm bounding box
                    img = Image.open(img_path)
                    w, h = img.size
                    boxes = [{"label": cls_name, "conf": 1.0, "xyxy": [0, 0, w, h]}]

                # Lấy box có confidence cao nhất
                best = max(boxes, key=lambda b: b["conf"])

                # Lưu ra file XML (Pascal VOC format — dùng được với Faster R-CNN)
                save_xml(img_path, best, cls_name)
                annotated += 1

            total_annotated += annotated
            total_skipped += skipped
            print(f"  ✅ Annotated: {annotated}  |  Bỏ qua (đã có): {skipped}")

    print("\n=== HOÀN TẤT ===")
    print(f"Tổng annotated: {total_annotated} ảnh")
    print(f"Tổng bỏ qua  : {total_skipped} ảnh (đã có XML từ trước)")
    print("Các file .xml đã được lưu cùng thư mục với ảnh.")


def save_xml(img_path: Path, box: dict, folder_label: str):
    img = Image.open(img_path)
    w, h = img.size
    x1, y1, x2, y2 = box["xyxy"]

    # Clamp trong giới hạn ảnh
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    root = ET.Element("annotation")
    ET.SubElement(root, "folder").text = img_path.parent.name
    ET.SubElement(root, "filename").text = img_path.name

    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(w)
    ET.SubElement(size, "height").text = str(h)
    ET.SubElement(size, "depth").text = "3"

    obj = ET.SubElement(root, "object")
    ET.SubElement(obj, "name").text = folder_label  # dùng nhãn từ thư mục, không từ YOLO
    ET.SubElement(obj, "confidence").text = f"{box['conf']:.3f}"

    bndbox = ET.SubElement(obj, "bndbox")
    ET.SubElement(bndbox, "xmin").text = str(x1)
    ET.SubElement(bndbox, "ymin").text = str(y1)
    ET.SubElement(bndbox, "xmax").text = str(x2)
    ET.SubElement(bndbox, "ymax").text = str(y2)

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(img_path.with_suffix(".xml"), encoding="utf-8", xml_declaration=True)


if __name__ == "__main__":
    print("=== TỰ ĐỘNG ANNOTATE BOUNDING BOX BẰNG YOLO ===")
    print("Format: Pascal VOC XML (dùng được với Faster R-CNN)\n")
    annotate_dataset(conf=0.3)
