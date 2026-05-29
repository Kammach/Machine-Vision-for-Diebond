import argparse
import csv
import random
import re
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# ==========================================================
# CONFIG
# ==========================================================

IMAGE_EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff"
}

DEFAULT_WIDTH = 5


# ==========================================================
# BASIC UTILS
# ==========================================================

def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS


def ask_yes_no(prompt: str, default: bool = False) -> bool:

    suffix = " [Y/n]: " if default else " [y/N]: "

    while True:

        ans = input(prompt + suffix).strip().lower()

        if not ans:
            return default

        if ans in {"y", "yes", "1", "true", "t", "ใช่"}:
            return True

        if ans in {"n", "no", "0", "false", "f", "ไม่"}:
            return False

        print("พิมพ์ y หรือ n")


# ==========================================================
# FILE LIST
# ==========================================================

def list_image_files(images_dir: Path) -> List[Path]:

    if not images_dir.exists():
        return []

    return sorted(
        [
            p for p in images_dir.iterdir()
            if is_image_file(p)
        ],
        key=lambda p: p.name.lower()
    )


def list_label_files(
    labels_dir: Path,
    recursive: bool = True
) -> List[Path]:

    if not labels_dir.exists():
        return []

    if recursive:
        return sorted(
            [
                p for p in labels_dir.rglob("*.txt")
                if p.is_file()
            ],
            key=lambda p: str(p).lower()
        )

    return sorted(
        [
            p for p in labels_dir.glob("*.txt")
            if p.is_file()
        ],
        key=lambda p: p.name.lower()
    )


# ==========================================================
# PARSER
# ==========================================================

def parse_class_mapping(raw: str) -> Dict[str, str]:
    """
    examples:
      5:6
      5=6,2=0
      5->6;1->3
    """

    if not raw.strip():
        return {}

    normalized = (
        raw
        .replace("->", ":")
        .replace("=", ":")
        .replace(";", ",")
    )

    mapping = {}

    for item in normalized.split(","):

        item = item.strip()

        if not item:
            continue

        if ":" not in item:
            raise ValueError(f"รูปแบบ mapping ผิด: {item}")

        old, new = item.split(":", 1)

        old = old.strip()
        new = new.strip()

        if not old.isdigit() or not new.isdigit():
            raise ValueError(f"class id ต้องเป็นเลข: {item}")

        mapping[old] = new

    return mapping


# ==========================================================
# AUTO DISCOVER DATASET
# ==========================================================

def discover_dataset_folders(root: Path) -> List[Path]:

    root = Path(root)

    if not root.exists():
        raise FileNotFoundError(f"ไม่พบ path: {root}")

    found = []

    for item in root.iterdir():

        if not item.is_dir():
            continue

        # กัน ALL ซ้ำ
        if item.name.upper() == "ALL":
            continue

        images_dir = item / "images"
        labels_dir = item / "labels"

        if images_dir.exists() and labels_dir.exists():
            found.append(item)

    return sorted(
        found,
        key=lambda p: p.name.lower()
    )


# ==========================================================
# BACKUP
# ==========================================================

def backup_dataset(
    source_root: Path,
    backup_root: Optional[Path] = None
) -> Path:

    source_root = Path(source_root)

    if not source_root.exists():
        raise FileNotFoundError(source_root)

    if backup_root is None:
        backup_root = (
            source_root.parent /
            f"{source_root.name}_backup_{timestamp()}"
        )

    if backup_root.exists():
        raise FileExistsError(backup_root)

    print(f"\nกำลัง backup...")
    print(source_root)
    print("->")
    print(backup_root)

    shutil.copytree(source_root, backup_root)

    print("backup เสร็จ")

    return backup_root


# ==========================================================
# PREPARE OUTPUT
# ==========================================================

def prepare_output_root(
    output_root: Path,
    clean: bool = False
) -> Tuple[Path, Path]:

    images_out = output_root / "images"
    labels_out = output_root / "labels"

    if output_root.exists():

        has_content = any(output_root.iterdir())

        if has_content:

            if clean:

                shutil.rmtree(output_root)

            else:

                if ask_yes_no(
                    f"{output_root} มีข้อมูลอยู่ ลบทิ้งไหม?",
                    default=False
                ):
                    shutil.rmtree(output_root)
                else:
                    print("ยกเลิก")
                    sys.exit(0)

    ensure_dir(images_out)
    ensure_dir(labels_out)

    return images_out, labels_out


# ==========================================================
# COPY EMPTY LABEL
# ==========================================================

def copy_empty_file(path: Path) -> None:

    ensure_dir(path.parent)

    path.write_text(
        "",
        encoding="utf-8"
    )


# ==========================================================
# COUNT TOTAL IMAGES
# ==========================================================

def count_total_images(
    sources: Sequence[Path]
) -> int:

    total = 0

    for src in sources:
        total += len(
            list_image_files(src / "images")
        )

    return total


# ==========================================================
# MERGE DATASET
# ==========================================================

def merge_datasets(
    sources: Sequence[Path],
    output_root: Path,
    clean_output: bool = False,
    create_empty_label_when_missing: bool = True,
    start_index: int = 0,
    width: Optional[int] = None,
) -> None:

    sources = [Path(s) for s in sources]

    for src in sources:

        if not src.exists():
            raise FileNotFoundError(src)

        if not (src / "images").exists():
            raise FileNotFoundError(
                f"ไม่พบ images ใน {src}"
            )

        if not (src / "labels").exists():
            raise FileNotFoundError(
                f"ไม่พบ labels ใน {src}"
            )

    total_images = count_total_images(sources)

    if width is None:

        width = max(
            DEFAULT_WIDTH,
            len(str(max(
                start_index + total_images - 1,
                0
            )))
        )

    images_out, labels_out = prepare_output_root(
        output_root,
        clean=clean_output
    )

    manifest_path = (
        output_root / "merge_manifest.csv"
    )

    print("\n================================================")
    print("เริ่มรวม dataset")
    print("================================================")
    print(f"จำนวน source : {len(sources)}")
    print(f"จำนวนภาพรวม : {total_images}")
    print(f"เริ่ม index  : {start_index}")
    print(f"digit width  : {width}")
    print("================================================")

    current_index = start_index

    copied_images = 0
    missing_labels = 0

    with open(
        manifest_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as mf:

        writer = csv.writer(mf)

        writer.writerow([
            "index",
            "source_folder",
            "source_image",
            "source_label",
            "output_image",
            "output_label",
        ])

        for src in sources:

            src_name = src.name

            images_dir = src / "images"
            labels_dir = src / "labels"

            images = list_image_files(images_dir)

            print(f"\n[{src_name}]")
            print(f"images = {len(images)}")

            for img_path in images:

                stem = img_path.stem

                label_path = (
                    labels_dir / f"{stem}.txt"
                )

                new_stem = (
                    f"img_{current_index:0{width}d}"
                )

                dst_img = (
                    images_out /
                    f"{new_stem}{img_path.suffix.lower()}"
                )

                dst_lbl = (
                    labels_out /
                    f"{new_stem}.txt"
                )

                shutil.copy2(
                    img_path,
                    dst_img
                )

                if label_path.exists():

                    shutil.copy2(
                        label_path,
                        dst_lbl
                    )

                else:

                    missing_labels += 1

                    if create_empty_label_when_missing:
                        copy_empty_file(dst_lbl)

                writer.writerow([
                    current_index,
                    src_name,
                    str(img_path.relative_to(src)),
                    str(label_path.relative_to(src))
                    if label_path.exists()
                    else "",
                    str(dst_img.relative_to(output_root)),
                    str(dst_lbl.relative_to(output_root)),
                ])

                current_index += 1
                copied_images += 1

    print("\n================================================")
    print("MERGE เสร็จ")
    print("================================================")
    print(f"copy images : {copied_images}")
    print(f"missing txt : {missing_labels}")
    print(f"manifest    : {manifest_path}")
    print("================================================")


# ==========================================================
# RELABEL
# ==========================================================

def relabel_lines(
    lines: Iterable[str],
    mapping: Dict[str, str]
) -> Tuple[List[str], int]:

    out_lines = []

    changed_count = 0

    for raw_line in lines:

        line = raw_line.strip()

        if not line:
            continue

        parts = line.split()

        if not parts:
            continue

        cls = parts[0]

        if cls in mapping:

            parts[0] = mapping[cls]

            changed_count += 1

        out_lines.append(
            " ".join(parts)
        )

    return out_lines, changed_count


def relabel_dataset(
    labels_root: Path,
    mapping: Dict[str, str],
    recursive: bool = True,
    backup_first: bool = False,
) -> None:

    labels_root = Path(labels_root)

    if not labels_root.exists():
        raise FileNotFoundError(labels_root)

    txt_files = list_label_files(
        labels_root,
        recursive=recursive
    )

    if not txt_files:
        print("ไม่พบ txt")
        return

    if backup_first:
        backup_dataset(labels_root.parent)

    changed_files = 0
    changed_lines = 0

    print("\n================================================")
    print("เริ่ม RELABEL")
    print("================================================")
    print(mapping)
    print("================================================")

    for txt in txt_files:

        try:

            original = txt.read_text(
                encoding="utf-8",
                errors="ignore"
            ).splitlines()

        except Exception as e:

            print(f"[ข้าม] {txt}")
            print(e)

            continue

        new_lines, changed = relabel_lines(
            original,
            mapping
        )

        if changed > 0:

            txt.write_text(
                "\n".join(new_lines) + "\n",
                encoding="utf-8"
            )

            changed_files += 1
            changed_lines += changed

    print("\n================================================")
    print("RELABEL เสร็จ")
    print("================================================")
    print(f"files changed : {changed_files}")
    print(f"lines changed : {changed_lines}")
    print("================================================")


# ==========================================================
# REMOVE CLASS
# ==========================================================

def remove_class(
    labels_root: Path,
    target_class: int
) -> None:

    txt_files = list_label_files(labels_root)

    removed = 0

    for txt in txt_files:

        lines = txt.read_text(
            encoding="utf-8",
            errors="ignore"
        ).splitlines()

        out = []

        changed = False

        for line in lines:

            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if parts[0] == str(target_class):

                removed += 1
                changed = True
                continue

            out.append(line)

        if changed:

            txt.write_text(
                "\n".join(out) + "\n",
                encoding="utf-8"
            )

    print(f"\nลบ class {target_class} เสร็จ")
    print(f"removed objects = {removed}")


# ==========================================================
# VALIDATE
# ==========================================================

def validate_dataset(
    dataset_root: Path,
    recursive_labels: bool = False
) -> None:

    dataset_root = Path(dataset_root)

    images_dir = dataset_root / "images"
    labels_dir = dataset_root / "labels"

    if not images_dir.exists():
        raise FileNotFoundError(images_dir)

    if not labels_dir.exists():
        raise FileNotFoundError(labels_dir)

    images = list_image_files(images_dir)

    labels = list_label_files(
        labels_dir,
        recursive=recursive_labels
    )

    image_stems = {
        p.stem for p in images
    }

    label_stems = {
        p.stem for p in labels
    }

    missing_labels = sorted(
        image_stems - label_stems
    )

    missing_images = sorted(
        label_stems - image_stems
    )

    class_counter = Counter()

    invalid_lines = []

    empty_labels = 0

    for label_file in labels:

        try:

            content = label_file.read_text(
                encoding="utf-8",
                errors="ignore"
            ).splitlines()

        except Exception as e:

            invalid_lines.append(
                (str(label_file), str(e))
            )

            continue

        if not content:
            empty_labels += 1

        for i, raw in enumerate(content, start=1):

            line = raw.strip()

            if not line:
                continue

            parts = line.split()

            if not parts:
                continue

            cls = parts[0]

            if not re.fullmatch(
                r"-?\d+",
                cls
            ):
                invalid_lines.append(
                    (
                        str(label_file),
                        f"line {i}: invalid class"
                    )
                )
                continue

            class_counter[int(cls)] += 1

    print("\n================================================")
    print("VALIDATE")
    print("================================================")
    print(f"images           : {len(images)}")
    print(f"labels           : {len(labels)}")
    print(f"missing labels   : {len(missing_labels)}")
    print(f"missing images   : {len(missing_images)}")
    print(f"empty labels     : {empty_labels}")
    print(f"invalid lines    : {len(invalid_lines)}")
    print("================================================")

    if class_counter:

        print("\nCLASS COUNT")

        for cls, cnt in sorted(class_counter.items()):
            print(f"class {cls} : {cnt}")

    if missing_labels:

        print("\nตัวอย่าง missing labels")

        for name in missing_labels[:10]:
            print(name)

    if missing_images:

        print("\nตัวอย่าง missing images")

        for name in missing_images[:10]:
            print(name)


# ==========================================================
# STATS
# ==========================================================

def show_stats(dataset_root: Path) -> None:
    validate_dataset(dataset_root)


# ==========================================================
# SPLIT DATASET
# ==========================================================

def split_dataset(
    dataset_root: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42
) -> None:

    random.seed(seed)

    dataset_root = Path(dataset_root)

    images_dir = dataset_root / "images"
    labels_dir = dataset_root / "labels"

    images = list_image_files(images_dir)

    random.shuffle(images)

    total = len(images)

    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)

    train_imgs = images[:train_count]
    val_imgs = images[train_count:train_count + val_count]
    test_imgs = images[train_count + val_count:]

    splits = {
        "train": train_imgs,
        "val": val_imgs,
        "test": test_imgs,
    }

    for split_name, split_images in splits.items():

        split_img_dir = dataset_root / split_name / "images"
        split_lbl_dir = dataset_root / split_name / "labels"

        ensure_dir(split_img_dir)
        ensure_dir(split_lbl_dir)

        for img_path in split_images:

            lbl_path = (
                labels_dir / f"{img_path.stem}.txt"
            )

            shutil.copy2(
                img_path,
                split_img_dir / img_path.name
            )

            if lbl_path.exists():

                shutil.copy2(
                    lbl_path,
                    split_lbl_dir / lbl_path.name
                )

    print("\nSPLIT เสร็จ")
    print(f"train = {len(train_imgs)}")
    print(f"val   = {len(val_imgs)}")
    print(f"test  = {len(test_imgs)}")


# ==========================================================
# MENU
# ==========================================================

def print_menu() -> None:

    print("\n================================================")
    print("YOLO DATASET TOOL")
    print("================================================")
    print("1) Merge Dataset")
    print("2) Relabel Class")
    print("3) Remove Class")
    print("4) Validate Dataset")
    print("5) Show Statistics")
    print("6) Backup Dataset")
    print("7) Split Train/Val/Test")
    print("0) Exit")
    print("================================================")


# ==========================================================
# INTERACTIVE
# ==========================================================

def run_interactive() -> None:

    while True:

        print_menu()

        choice = input("เลือกเมนู: ").strip()

        try:

            # ======================================================
            # MERGE
            # ======================================================

            if choice == "1":

                dataset_root = Path(
                    input(
                        "ใส่ root dataset path:\n> "
                    ).strip()
                )

                sources = discover_dataset_folders(
                    dataset_root
                )

                if not sources:
                    print("ไม่พบ dataset")
                    continue

                print("\nพบ dataset:")

                for s in sources:
                    print(" -", s.name)

                output_root = Path(
                    input(
                        "ใส่ output root [ALL]: "
                    ).strip()
                    or "ALL"
                )

                clean = ask_yes_no(
                    "ลบ output เดิมไหม?",
                    default=False
                )

                merge_datasets(
                    sources=sources,
                    output_root=output_root,
                    clean_output=clean
                )

            # ======================================================
            # RELABEL
            # ======================================================

            elif choice == "2":

                labels_root = Path(
                    input(
                        "ใส่ labels folder:\n> "
                    ).strip()
                )

                raw_map = input(
                    "ใส่ mapping เช่น 5:6,1:0\n> "
                ).strip()

                mapping = parse_class_mapping(
                    raw_map
                )

                backup = ask_yes_no(
                    "backup ก่อนแก้ไหม?",
                    default=True
                )

                relabel_dataset(
                    labels_root,
                    mapping,
                    recursive=True,
                    backup_first=backup
                )

            # ======================================================
            # REMOVE CLASS
            # ======================================================

            elif choice == "3":

                labels_root = Path(
                    input(
                        "ใส่ labels folder:\n> "
                    ).strip()
                )

                target_class = int(
                    input(
                        "ใส่ class ที่ต้องการลบ: "
                    ).strip()
                )

                remove_class(
                    labels_root,
                    target_class
                )

            # ======================================================
            # VALIDATE
            # ======================================================

            elif choice == "4":

                dataset_root = Path(
                    input(
                        "ใส่ dataset root:\n> "
                    ).strip()
                )

                validate_dataset(dataset_root)

            # ======================================================
            # STATS
            # ======================================================

            elif choice == "5":

                dataset_root = Path(
                    input(
                        "ใส่ dataset root:\n> "
                    ).strip()
                )

                show_stats(dataset_root)

            # ======================================================
            # BACKUP
            # ======================================================

            elif choice == "6":

                source_root = Path(
                    input(
                        "ใส่ dataset root:\n> "
                    ).strip()
                )

                backup_dataset(source_root)

            # ======================================================
            # SPLIT
            # ======================================================

            elif choice == "7":

                dataset_root = Path(
                    input(
                        "ใส่ dataset root:\n> "
                    ).strip()
                )

                split_dataset(dataset_root)

            # ======================================================
            # EXIT
            # ======================================================

            elif choice == "0":

                print("จบการทำงาน")
                return

            else:

                print("เลือกไม่ถูก")

        except Exception as e:

            print(f"\n[ERROR]")
            print(e)


# ==========================================================
# MAIN
# ==========================================================

def main():

    run_interactive()


if __name__ == "__main__":
    main()