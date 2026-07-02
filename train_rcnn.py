import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import torch
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Force UTF-8 output on Windows
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

# ── Config ────────────────────────────────────────────────
CLASSES     = ["__background__", "Bus", "Car", "Truck"]
NUM_CLASSES = len(CLASSES)
EPOCHS      = 10
BATCH_SIZE  = 2
LR          = 1e-3
DEVICE      = torch.device("cpu")
MODEL_PATH  = "models/best_model_rcnn.pth"
# ──────────────────────────────────────────────────────────


class VehicleDataset(Dataset):
    def __init__(self, root: str):
        self.samples = []
        root = Path(root)
        for cls_name in ["Bus", "Car", "Truck"]:
            cls_dir = root / cls_name
            if not cls_dir.exists():
                continue
            for xml_path in cls_dir.glob("*.xml"):
                img_path = xml_path.with_suffix(".jpg")
                if not img_path.exists():
                    for ext in [".jpeg", ".png", ".bmp", ".webp"]:
                        alt = xml_path.with_suffix(ext)
                        if alt.exists():
                            img_path = alt
                            break
                    else:
                        continue
                self.samples.append((img_path, xml_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, xml_path = self.samples[idx]

        img = Image.open(img_path).convert("RGB")
        img_tensor = F.to_tensor(img)

        tree = ET.parse(xml_path)
        root = tree.getroot()

        boxes, labels = [], []

        for obj in root.findall("object"):
            cls_name = obj.find("name").text
            if cls_name not in CLASSES:
                continue
            label = CLASSES.index(cls_name)

            bb = obj.find("bndbox")
            x1 = float(bb.find("xmin").text)
            y1 = float(bb.find("ymin").text)
            x2 = float(bb.find("xmax").text)
            y2 = float(bb.find("ymax").text)

            if x2 <= x1 or y2 <= y1:
                continue

            boxes.append([x1, y1, x2, y2])
            labels.append(label)

        if not boxes:
            _, h, w = img_tensor.shape
            boxes  = [[0, 0, w, h]]
            labels = [CLASSES.index(xml_path.parent.name)]

        target = {
            "boxes":  torch.tensor(boxes,  dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }

        return img_tensor, target


def build_model(num_classes: int):
    model = fasterrcnn_mobilenet_v3_large_fpn(weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def collate_fn(batch):
    return tuple(zip(*batch))


def train():
    train_dataset = VehicleDataset("data/train")
    val_dataset   = VehicleDataset("data/val")

    print(f"[INFO] Train: {len(train_dataset)} images  |  Val: {len(val_dataset)} images")

    if len(train_dataset) == 0:
        print("[ERROR] No images with XML found. Run auto_annotate.py first.")
        return

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE,
                              shuffle=False,  collate_fn=collate_fn)

    model = build_model(NUM_CLASSES).to(DEVICE)

    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, momentum=0.9, weight_decay=1e-4
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    os.makedirs("models", exist_ok=True)

    best_val_loss = float("inf")
    train_losses, val_losses = [], []

    print(f"\n=== TRAINING Faster R-CNN ({EPOCHS} epochs, device={DEVICE}) ===\n")

    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        epoch_loss = 0
        for imgs, targets in train_loader:
            imgs    = [img.to(DEVICE) for img in imgs]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            loss_dict = model(imgs, targets)
            loss = sum(loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.train()
        val_loss = 0
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs    = [img.to(DEVICE) for img in imgs]
                targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
                loss_dict = model(imgs, targets)
                val_loss += sum(loss_dict.values()).item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        lr_scheduler.step()

        print(f"Epoch [{epoch:2}/{EPOCHS}]  train_loss={avg_train_loss:.4f}  val_loss={avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"           -> Saved best model (val_loss={best_val_loss:.4f})")

    # Plot
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses,   label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Faster R-CNN Training")
    plt.legend()
    plt.tight_layout()
    plt.savefig("ket_qua_training_rcnn.png")
    print("\n-> Chart saved: ket_qua_training_rcnn.png")
    print(f"-> Best model : {MODEL_PATH}")


if __name__ == "__main__":
    train()
