import os
import random
import shutil

random.seed(42)

LIMITS = {
    "train": 300,
    "valid": 40,
    "test": 20
}

DATASET = "dataset"

for folder, limit in LIMITS.items():

    folder_path = os.path.join(DATASET, folder)

    for cls in os.listdir(folder_path):

        cls_path = os.path.join(folder_path, cls)

        if not os.path.isdir(cls_path):
            continue

        images = [
            f for f in os.listdir(cls_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        if len(images) <= limit:
            print(f"{cls}: {len(images)} images (OK)")
            continue

        random.shuffle(images)

        delete_images = images[limit:]

        for img in delete_images:
            os.remove(os.path.join(cls_path, img))

        print(f"{cls}: kept {limit} images")

print("\nDataset berhasil diperkecil.")