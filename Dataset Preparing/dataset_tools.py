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
# CONFIGURATION
# ==========================================================

IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"
}

DEFAULT_WIDTH = 5  # Default zero-padding width for image numbering (e.g., img_00001)


# ==========================================================
# BASIC UTILITIES
# ==========================================================

def timestamp() -> str:
    """Return current timestamp in format: YYYYMMDD_HHMMSS"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: Path) -> None:
    """Create directory if it doesn't exist (including parents)"""
    path.mkdir(parents=True, exist_ok=True)


def is_image_file(path: Path) -> bool:
    """Check if file is a supported image format"""
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS


def ask_yes_no(prompt: str, default: bool = False) -> bool:
    """Interactive yes/no prompt with Thai + English support"""
    suffix = " [Y/n]: " if default else " [y/N]: "

    while True:
        ans = input(prompt + suffix).strip().lower()

        if not ans:
            return default

        if ans in {"y", "yes", "1", "true", "t", "ใช่"}:
            return True

        if ans in {"n", "no", "0", "false", "f", "ไม่"}:
            return False

        print("Please type y or n")


# ==========================================================
# FILE LISTING UTILITIES
# ==========================================================

def list_image_files(images_dir: Path) -> List[Path]:
    """List all image files in directory, sorted by name"""
    if not images_dir.exists():
        return []

    return sorted(
        [p for p in images_dir.iterdir() if is_image_file(p)],
        key=lambda p: p.name.lower()
    )


def list_label_files(labels_dir: Path, recursive: bool = True) -> List[Path]:
    """List all .txt label files (support recursive search)"""
    if not labels_dir.exists():
        return []

    if recursive:
        return sorted(
            [p for p in labels_dir.rglob("*.txt") if p.is_file()],
            key=lambda p: str(p).lower()
        )

    return sorted(
        [p for p in labels_dir.glob("*.txt") if p.is_file()],
        key=lambda p: p.name.lower()
    )


# ==========================================================
# CLASS MAPPING PARSER
# ==========================================================

def parse_class_mapping(raw: str) -> Dict[str, str]:
    """
    Parse class mapping string.
    Examples:
        "5:6"           -> { "5": "6" }
        "5=6,2=0"       -> { "5": "6", "2": "0" }
        "5->6;1->3"     -> { "5": "6", "1": "3" }
    """
    if not raw.strip():
        return {}

    # Normalize different separators
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
            raise ValueError(f"Invalid mapping format: {item}")

        old, new = item.split(":", 1)
        old = old.strip()
        new = new.strip()

        if not old.isdigit() or not new.isdigit():
            raise ValueError(f"Class IDs must be numbers: {item}")

        mapping[old] = new

    return mapping


# ==========================================================
# DATASET DISCOVERY
# ==========================================================

def discover_dataset_folders(root: Path) -> List[Path]:
    """
    Auto-discover dataset folders that contain 'images' and 'labels' subfolders.
    Skips folder named 'ALL'.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Path not found: {root}")

    found = []

    for item in root.iterdir():
        if not item.is_dir():
            continue
        if item.name.upper() == "ALL":
            continue

        if (item / "images").exists() and (item / "labels").exists():
            found.append(item)

    return sorted(found, key=lambda p: p.name.lower())


# ==========================================================
# BACKUP
# ==========================================================

def backup_dataset(source_root: Path, backup_root: Optional[Path] = None) -> Path:
    """Create backup of entire dataset"""
    source_root = Path(source_root)

    if not source_root.exists():
        raise FileNotFoundError(source_root)

    if backup_root is None:
        backup_root = source_root.parent / f"{source_root.name}_backup_{timestamp()}"

    if backup_root.exists():
        raise FileExistsError(f"Backup folder already exists: {backup_root}")

    print(f"\nCreating backup...")
    print(f"From : {source_root}")
    print(f"To   : {backup_root}")

    shutil.copytree(source_root, backup_root)

    print("✅ Backup completed")
    return backup_root


# ==========================================================
# OUTPUT PREPARATION
# ==========================================================

def prepare_output_root(output_root: Path, clean: bool = False) -> Tuple[Path, Path]:
    """Prepare output directories (images & labels)"""
    images_out = output_root / "images"
    labels_out = output_root / "labels"

    if output_root.exists() and any(output_root.iterdir()):
        if clean or ask_yes_no(f"{output_root} already exists. Delete it?", default=False):
            shutil.rmtree(output_root)
        else:
            print("Operation cancelled by user.")
            sys.exit(0)

    ensure_dir(images_out)
    ensure_dir(labels_out)

    return images_out, labels_out


# ==========================================================
# HELPER FUNCTIONS
# ==========================================================

def copy_empty_file(path: Path) -> None:
    """Create an empty label file"""
    ensure_dir(path.parent)
    path.write_text("", encoding="utf-8")


def count_total_images(sources: Sequence[Path]) -> int:
    """Count total images across all source datasets"""
    total = 0
    for src in sources:
        total += len(list_image_files(src / "images"))
    return total


# ==========================================================
# MERGE DATASETS
# ==========================================================

def merge_datasets(
    sources: Sequence[Path],
    output_root: Path,
    clean_output: bool = False,
    create_empty_label_when_missing: bool = True,
    start_index: int = 0,
    width: Optional[int] = None,
) -> None:
    """
    Merge multiple YOLO datasets into one with sequential naming.
    Example output: img_00001.jpg + img_00001.txt
    """
    sources = [Path(s) for s in sources]

    # Validation
    for src in sources:
        if not (src / "images").exists():
            raise FileNotFoundError(f"Missing 'images' folder in {src}")
        if not (src / "labels").exists():
            raise FileNotFoundError(f"Missing 'labels' folder in {src}")

    total_images = count_total_images(sources)

    if width is None:
        width = max(DEFAULT_WIDTH, len(str(start_index + total_images - 1)))

    images_out, labels_out = prepare_output_root(output_root, clean=clean_output)

    manifest_path = output_root / "merge_manifest.csv"

    print("\n" + "="*60)
    print("STARTING DATASET MERGE")
    print("="*60)
    print(f"Source folders : {len(sources)}")
    print(f"Total images   : {total_images}")
    print(f"Start index    : {start_index}")
    print(f"Filename width : {width}")
    print("="*60)

    current_index = start_index
    copied_images = 0
    missing_labels = 0

    with open(manifest_path, "w", newline="", encoding="utf-8") as mf:
        writer = csv.writer(mf)
        writer.writerow([
            "index", "source_folder", "source_image", "source_label",
            "output_image", "output_label"
        ])

        for src in sources:
            src_name = src.name
            images = list_image_files(src / "images")

            print(f"\n[{src_name}] - {len(images)} images")

            for img_path in images:
                stem = img_path.stem
                label_path = (src / "labels") / f"{stem}.txt"

                new_stem = f"img_{current_index:0{width}d}"

                dst_img = images_out / f"{new_stem}{img_path.suffix.lower()}"
                dst_lbl = labels_out / f"{new_stem}.txt"

                # Copy image
                shutil.copy2(img_path, dst_img)

                # Copy label or create empty
                if label_path.exists():
                    shutil.copy2(label_path, dst_lbl)
                else:
                    missing_labels += 1
                    if create_empty_label_when_missing:
                        copy_empty_file(dst_lbl)

                # Write to manifest
                writer.writerow([
                    current_index,
                    src_name,
                    str(img_path.relative_to(src)),
                    str(label_path.relative_to(src)) if label_path.exists() else "",
                    str(dst_img.relative_to(output_root)),
                    str(dst_lbl.relative_to(output_root)),
                ])

                current_index += 1
                copied_images += 1

    print("\n" + "="*60)
    print("MERGE COMPLETED SUCCESSFULLY")
    print("="*60)
    print(f"Images copied     : {copied_images}")
    print(f"Missing labels    : {missing_labels}")
    print(f"Manifest file     : {manifest_path}")
    print("="*60)


# ==========================================================
# RELABEL CLASSES
# ==========================================================

def relabel_lines(lines: Iterable[str], mapping: Dict[str, str]) -> Tuple[List[str], int]:
    """Apply class ID mapping to label lines"""
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

        out_lines.append(" ".join(parts))

    return out_lines, changed_count


def relabel_dataset(
    labels_root: Path,
    mapping: Dict[str, str],
    recursive: bool = True,
    backup_first: bool = False,
) -> None:
    """Change class IDs in all label files according to mapping"""
    labels_root = Path(labels_root)

    if not labels_root.exists():
        raise FileNotFoundError(labels_root)

    txt_files = list_label_files(labels_root, recursive=recursive)

    if not txt_files:
        print("No label files found.")
        return

    if backup_first:
        backup_dataset(labels_root.parent)

    changed_files = 0
    changed_lines = 0

    print("\n" + "="*60)
    print("STARTING RELABEL OPERATION")
    print("="*60)
    print(f"Mapping: {mapping}")
    print("="*60)

    for txt in txt_files:
        try:
            original = txt.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception as e:
            print(f"[Skipped] {txt} - {e}")
            continue

        new_lines, changed = relabel_lines(original, mapping)

        if changed > 0:
            txt.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
            changed_files += 1
            changed_lines += changed

    print("\n" + "="*60)
    print("RELABEL COMPLETED")
    print("="*60)
    print(f"Files changed : {changed_files}")
    print(f"Lines changed : {changed_lines}")
    print("="*60)


# ==========================================================
# REMOVE SPECIFIC CLASS
# ==========================================================

def remove_class(labels_root: Path, target_class: int) -> None:
    """Remove all annotations of a specific class from dataset"""
    txt_files = list_label_files(labels_root)
    removed = 0

    for txt in txt_files:
        lines = txt.read_text(encoding="utf-8", errors="ignore").splitlines()
        out = []
        changed = False

        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if parts and parts[0] == str(target_class):
                removed += 1
                changed = True
                continue
            out.append(line)

        if changed:
            txt.write_text("\n".join(out) + "\n", encoding="utf-8")

    print(f"\n✅ Removed class {target_class}")
    print(f"Total objects removed: {removed}")


# ==========================================================
# VALIDATE & STATISTICS
# ==========================================================

def validate_dataset(dataset_root: Path, recursive_labels: bool = False) -> None:
    """Validate dataset integrity and show statistics"""
    dataset_root = Path(dataset_root)
    images_dir = dataset_root / "images"
    labels_dir = dataset_root / "labels"

    if not images_dir.exists():
        raise FileNotFoundError(f"Images folder not found: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels folder not found: {labels_dir}")

    images = list_image_files(images_dir)
    labels = list_label_files(labels_dir, recursive=recursive_labels)

    image_stems = {p.stem for p in images}
    label_stems = {p.stem for p in labels}

    missing_labels = sorted(image_stems - label_stems)
    missing_images = sorted(label_stems - image_stems)

    class_counter = Counter()
    invalid_lines = []
    empty_labels = 0

    for label_file in labels:
        try:
            content = label_file.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception as e:
            invalid_lines.append((str(label_file), str(e)))
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
            if not re.fullmatch(r"-?\d+", cls):
                invalid_lines.append((str(label_file), f"line {i}: invalid class"))
                continue

            class_counter[int(cls)] += 1

    print("\n" + "="*60)
    print("DATASET VALIDATION REPORT")
    print("="*60)
    print(f"Images           : {len(images)}")
    print(f"Labels           : {len(labels)}")
    print(f"Missing labels   : {len(missing_labels)}")
    print(f"Missing images   : {len(missing_images)}")
    print(f"Empty labels     : {empty_labels}")
    print(f"Invalid lines    : {len(invalid_lines)}")
    print("="*60)

    if class_counter:
        print("\nCLASS DISTRIBUTION")
        for cls, cnt in sorted(class_counter.items()):
            print(f"Class {cls:2d} : {cnt:6d} objects")

    if missing_labels:
        print(f"\nMissing labels (first 10):")
        for name in missing_labels[:10]:
            print(f"  • {name}")


def show_stats(dataset_root: Path) -> None:
    """Alias for validate_dataset"""
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
    """Split dataset into train/val/test folders"""
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
            lbl_path = labels_dir / f"{img_path.stem}.txt"

            shutil.copy2(img_path, split_img_dir / img_path.name)

            if lbl_path.exists():
                shutil.copy2(lbl_path, split_lbl_dir / lbl_path.name)

    print("\n✅ Dataset split completed")
    print(f"Train : {len(train_imgs)} images")
    print(f"Val   : {len(val_imgs)} images")
    print(f"Test  : {len(test_imgs)} images")


# ==========================================================
# MENU & INTERACTIVE INTERFACE
# ==========================================================

def print_menu() -> None:
    print("\n" + "="*60)
    print("YOLO DATASET TOOL")
    print("="*60)
    print("1) Merge Multiple Datasets")
    print("2) Relabel Classes")
    print("3) Remove Specific Class")
    print("4) Validate Dataset")
    print("5) Show Statistics")
    print("6) Backup Dataset")
    print("7) Split Train/Val/Test")
    print("0) Exit")
    print("="*60)


def run_interactive() -> None:
    """Main interactive menu loop"""
    while True:
        print_menu()
        choice = input("Enter menu number: ").strip()

        try:
            if choice == "1":
                # Merge
                root = Path(input("Enter root dataset path:\n> ").strip())
                sources = discover_dataset_folders(root)
                if not sources:
                    print("No datasets found.")
                    continue

                print("\nFound datasets:")
                for s in sources:
                    print(f"  • {s.name}")

                output = Path(input("Enter output folder [ALL]: ").strip() or "ALL")
                clean = ask_yes_no("Clear existing output folder?", default=False)

                merge_datasets(sources, output, clean_output=clean)

            elif choice == "2":
                # Relabel
                labels_root = Path(input("Enter labels folder:\n> ").strip())
                raw_map = input("Enter mapping (e.g. 5:6,2:0): ").strip()
                mapping = parse_class_mapping(raw_map)
                backup = ask_yes_no("Backup before modifying?", default=True)
                relabel_dataset(labels_root, mapping, backup_first=backup)

            elif choice == "3":
                # Remove class
                labels_root = Path(input("Enter labels folder:\n> ").strip())
                target = int(input("Enter class ID to remove: ").strip())
                remove_class(labels_root, target)

            elif choice == "4":
                # Validate
                root = Path(input("Enter dataset root:\n> ").strip())
                validate_dataset(root)

            elif choice == "5":
                # Stats
                root = Path(input("Enter dataset root:\n> ").strip())
                show_stats(root)

            elif choice == "6":
                # Backup
                root = Path(input("Enter dataset root:\n> ").strip())
                backup_dataset(root)

            elif choice == "7":
                # Split
                root = Path(input("Enter dataset root:\n> ").strip())
                split_dataset(root)

            elif choice == "0":
                print("Goodbye!")
                return
            else:
                print("Invalid selection.")

        except Exception as e:
            print(f"\n[ERROR] {e}")


# ==========================================================
# MAIN ENTRY POINT
# ==========================================================

def main():
    run_interactive()


if __name__ == "__main__":
    main()
