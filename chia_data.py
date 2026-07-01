import os
import shutil
import random
from pathlib import Path

# Tỷ lệ chia
TRAIN_RATIO = 0.8

# Thư mục chứa ảnh mới tải về (đặt ảnh vào đây trước khi chạy)
# Cấu trúc:
#   new_data/
#     Bus/   ← ảnh bus mới
#     Car/   ← ảnh car mới
#     Truck/ ← ảnh truck mới
NEW_DATA_DIR = "new_data"

CLASSES = ["Bus", "Car", "Truck"]
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def chia_data(new_data_dir: str, train_ratio: float = 0.8):
    new_data_dir = Path(new_data_dir)

    if not new_data_dir.exists():
        print(f"[LỖI] Không tìm thấy thư mục '{new_data_dir}'")
        print("Hãy tạo thư mục new_data/ và đặt ảnh vào các thư mục con Bus/, Car/, Truck/")
        return

    total_added = {"Bus": 0, "Car": 0, "Truck": 0}

    for cls in CLASSES:
        src_dir = new_data_dir / cls
        if not src_dir.exists():
            print(f"[SKIP] Không có thư mục {src_dir}, bỏ qua class {cls}")
            continue

        # Lấy danh sách ảnh hợp lệ
        images = [f for f in src_dir.iterdir() if f.suffix.lower() in EXTENSIONS]
        if not images:
            print(f"[SKIP] Không có ảnh nào trong {src_dir}")
            continue

        random.seed(42)
        random.shuffle(images)

        n_train = int(len(images) * train_ratio)
        train_images = images[:n_train]
        val_images = images[n_train:]

        # Tạo thư mục đích nếu chưa có
        train_dst = Path("data/train") / cls
        val_dst = Path("data/val") / cls
        train_dst.mkdir(parents=True, exist_ok=True)
        val_dst.mkdir(parents=True, exist_ok=True)

        # Lấy tên ảnh đã có để tránh ghi đè
        existing_train = {f.name for f in train_dst.iterdir()}
        existing_val = {f.name for f in val_dst.iterdir()}

        added_train = added_val = skipped = 0

        for img in train_images:
            if img.name in existing_train:
                skipped += 1
                continue
            shutil.copy2(img, train_dst / img.name)
            added_train += 1

        for img in val_images:
            if img.name in existing_val:
                skipped += 1
                continue
            shutil.copy2(img, val_dst / img.name)
            added_val += 1

        total_added[cls] = added_train + added_val
        print(f"[{cls}] Thêm {added_train} ảnh vào train, {added_val} ảnh vào val"
              + (f", bỏ qua {skipped} ảnh trùng tên" if skipped else ""))

    print("\n=== TỔNG KẾT ===")
    for cls in CLASSES:
        train_count = len(list((Path("data/train") / cls).iterdir()))
        val_count = len(list((Path("data/val") / cls).iterdir()))
        print(f"  {cls:6}: train={train_count}  val={val_count}  (thêm mới: {total_added[cls]})")
    print("\n✅ Xong! Chạy python train.py để retrain model.")


if __name__ == "__main__":
    chia_data(NEW_DATA_DIR, TRAIN_RATIO)
