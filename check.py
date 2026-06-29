import os
for c in ['Bus', 'Car', 'Truck']:
    path = f"data/train/{c}"
    if os.path.exists(path):
        print(f"{c} train:", len(os.listdir(path)))