import os
import random
import shutil

# =========================
# CONFIG
# =========================
SOURCE_IMAGES = "dataset_raw/images"
SOURCE_LABELS = "dataset_raw/labels"

OUTPUT_DIR = "dataset"

TRAIN_RATIO = 0.8
RANDOM_SEED = 42

# =========================
# CREATE FOLDERS
# =========================
def make_dirs():
    for split in ["train", "val"]:
        os.makedirs(os.path.join(OUTPUT_DIR, "images", split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, "labels", split), exist_ok=True)

# =========================
# GET FILE LIST
# =========================
def get_image_files():
    return [f for f in os.listdir(SOURCE_IMAGES)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))]

# =========================
# SPLIT DATA
# =========================
def split_dataset(files):
    random.seed(RANDOM_SEED)
    random.shuffle(files)

    split_idx = int(len(files) * TRAIN_RATIO)
    return files[:split_idx], files[split_idx:]

# =========================
# COPY FILES
# =========================
def copy_files(file_list, split):
    for img_file in file_list:
        name, _ = os.path.splitext(img_file)
        label_file = name + ".txt"

        src_img = os.path.join(SOURCE_IMAGES, img_file)
        src_lbl = os.path.join(SOURCE_LABELS, label_file)

        dst_img = os.path.join(OUTPUT_DIR, "images", split, img_file)
        dst_lbl = os.path.join(OUTPUT_DIR, "labels", split, label_file)

        # copy image
        shutil.copy2(src_img, dst_img)

        # copy label (if exists)
        if os.path.exists(src_lbl):
            shutil.copy2(src_lbl, dst_lbl)
        else:
            print(f"⚠️ Missing label for: {img_file}")

# =========================
# MAIN
# =========================
def main():
    make_dirs()

    files = get_image_files()
    print(f"Total images: {len(files)}")

    train_files, val_files = split_dataset(files)

    print(f"Train: {len(train_files)}")
    print(f"Val: {len(val_files)}")

    copy_files(train_files, "train")
    copy_files(val_files, "val")

    print("✅ Done splitting dataset!")

if __name__ == "__main__":
    main()