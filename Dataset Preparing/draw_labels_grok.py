import cv2
import os
import shutil

# ========================= CONFIGURATION =========================

IMG_DIR = r"D:\March\MCphase3\chip\dataset_raw\images"
LBL_DIR = r"D:\March\MCphase3\chip\dataset_raw\labels"

PROGRESS_FILE = "progress.txt"

# Class definition
CLASS_NAMES = {0: "NG", 1: "OK"}
current_class = 0                    # 0 = NG, 1 = OK
persistent_edit = False              # Keep edit mode across images
BOX_THICKNESS = 1

# Global variables for current image
current_img = None
current_boxes = []
edit_mode = False
img_w, img_h = 0, 0
current_label_path = ""
current_image_name = ""

# Mouse drawing variables
drawing = False
start_x, start_y = -1, -1
temp_box = None


# ========================= UTILITY FUNCTIONS =========================

def ensure_dirs():
    """Create images and labels directories if they don't exist"""
    os.makedirs(IMG_DIR, exist_ok=True)
    os.makedirs(LBL_DIR, exist_ok=True)


def normalize_dataset():
    """
    Rename all images to standardized format: img_0001.jpg, img_0002.jpg, ...
    This helps maintain consistent ordering and easy management.
    """
    print("\n🔄 Normalizing image filenames to img_XXXX format...")

    files = sorted([
        f for f in os.listdir(IMG_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    if not files:
        print("No image files found.")
        return

    print(f"Found {len(files)} image files.")

    # Step 1: Move to temporary names to avoid conflicts
    temp_list = []
    for i, old_name in enumerate(files):
        old_path = os.path.join(IMG_DIR, old_name)
        temp_name = f"temp_{i:04d}.tmp"
        temp_path = os.path.join(IMG_DIR, temp_name)

        try:
            shutil.move(old_path, temp_path)
            ext = os.path.splitext(old_name)[1].lower()
            temp_list.append((temp_name, ext))
        except Exception as e:
            print(f"❌ Error moving to temp: {old_name} → {e}")

    # Step 2: Rename to final standardized names
    for i, (temp_name, ext) in enumerate(temp_list, 1):
        temp_path = os.path.join(IMG_DIR, temp_name)
        new_name = f"img_{i:04d}{ext}"
        new_path = os.path.join(IMG_DIR, new_name)

        try:
            shutil.move(temp_path, new_path)
            print(f"   Renamed: {files[i-1]} → {new_name}")
        except Exception as e:
            print(f"❌ Error renaming: {temp_name} → {e}")

    print("✅ Filename normalization completed.\n")


# ========================= PROGRESS MANAGEMENT =========================

def load_progress() -> int:
    """Load last labeling progress from file"""
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, "r") as f:
                return int(f.read().strip())
        except:
            return 0
    return 0


def save_progress(idx: int):
    """Save current progress index"""
    with open(PROGRESS_FILE, "w") as f:
        f.write(str(idx))


# ========================= LABEL OPERATIONS =========================

def load_labels(label_path: str):
    """Load YOLO format labels and convert to pixel coordinates"""
    boxes = []
    if not os.path.exists(label_path):
        return boxes

    with open(label_path, "r") as f:
        for line in f.readlines():
            if not line.strip():
                continue
            try:
                cls, x, y, bw, bh = map(float, line.strip().split())
                cls = int(cls)

                # Safety: Fix invalid class
                if cls not in CLASS_NAMES:
                    print(f"⚠️ Invalid class {cls} in {os.path.basename(label_path)} → changed to 0 (NG)")
                    cls = 0

                # Convert YOLO normalized format to pixel coordinates
                x1 = int((x - bw / 2) * img_w)
                y1 = int((y - bh / 2) * img_h)
                x2 = int((x + bw / 2) * img_w)
                y2 = int((y + bh / 2) * img_h)

                boxes.append((x1, y1, x2, y2, cls))
            except:
                continue
    return boxes


def save_labels():
    """Save current boxes back to YOLO format label file"""
    with open(current_label_path, "w") as f:
        for (x1, y1, x2, y2, cls) in current_boxes:
            w = x2 - x1
            h = y2 - y1
            x_center = (x1 + w / 2) / img_w
            y_center = (y1 + h / 2) / img_h
            bw = w / img_w
            bh = h / img_h
            f.write(f"{cls} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}\n")

    print(f"✅ Saved: {os.path.basename(current_label_path)}")


# ========================= DRAWING & DISPLAY =========================

def draw() -> cv2.typing.MatLike:
    """Draw bounding boxes and information on the image"""
    img = current_img.copy()

    for (x1, y1, x2, y2, cls) in current_boxes:
        class_name = CLASS_NAMES.get(cls, f"UNK({cls})")
        color = (0, 0, 255) if cls == 0 else (0, 255, 0)  # Red=NG, Green=OK

        cv2.rectangle(img, (x1, y1), (x2, y2), color, BOX_THICKNESS)
        cv2.putText(img, class_name, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, max(2, BOX_THICKNESS))

    # Draw temporary box while dragging
    if temp_box:
        x1, y1, x2, y2 = temp_box
        color = (0, 0, 255) if current_class == 0 else (0, 255, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, BOX_THICKNESS)

    # Display information
    mode_text = "PERSISTENT EDIT" if persistent_edit else "EDIT MODE" if edit_mode else "VIEW MODE"
    cv2.putText(img, f"Image: {current_image_name}  |  Class: {CLASS_NAMES[current_class]}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    cv2.putText(img, f"Mode: {mode_text}  |  Boxes: {len(current_boxes)}  |  Thickness: {BOX_THICKNESS}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    cv2.putText(img, "Press H for Help", (10, img_h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    return img


# ========================= MOUSE CALLBACK =========================

def remove_nearest_box(x: int, y: int):
    """Remove the bounding box closest to the clicked position"""
    global current_boxes
    if not current_boxes:
        return

    min_idx = min(range(len(current_boxes)),
                  key=lambda i: ((current_boxes[i][0] + current_boxes[i][2]) // 2 - x) ** 2 +
                               ((current_boxes[i][1] + current_boxes[i][3]) // 2 - y) ** 2)
    current_boxes.pop(min_idx)
    print("🗑️ Removed nearest box")


def mouse_callback(event, x, y, flags, param):
    """Handle mouse events for drawing and deleting boxes"""
    global drawing, start_x, start_y, temp_box

    if not (edit_mode or persistent_edit):
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_x, start_y = x, y

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        temp_box = (start_x, start_y, x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if temp_box:
            x1 = max(0, min(start_x, x))
            y1 = max(0, min(start_y, y))
            x2 = min(img_w, max(start_x, x))
            y2 = min(img_h, max(start_y, y))

            if (x2 - x1) > 10 and (y2 - y1) > 10:
                current_boxes.append((x1, y1, x2, y2, current_class))
            temp_box = None

    elif event == cv2.EVENT_RBUTTONDOWN:
        remove_nearest_box(x, y)


def print_help():
    """Display keyboard shortcuts"""
    print("\n" + "=" * 70)
    print("                   YOLO LABELING TOOL - HELP")
    print("=" * 70)
    print("S     = Save + Next")
    print("N / 0 = Next (without saving)")
    print("B     = Back to previous image")
    print("E     = Toggle Edit Mode")
    print("P     = Toggle Persistent Edit")
    print("C     = Switch Class (NG ↔ OK)")
    print("D / Z = Delete last box")
    print("+ / =  = Increase box thickness")
    print("- / _  = Decrease box thickness")
    print("R     = Reset current image")
    print("H     = Show this help")
    print("Q     = Quit")
    print("Right Click = Delete nearest box")
    print("=" * 70 + "\n")


# ========================= MAIN PROGRAM =========================

ensure_dirs()
print_help()

# Ask user if they want to normalize filenames
reorganize = input("Do you want to normalize all image filenames to img_XXXX format before starting? (y/n): ").strip().lower()

if reorganize == 'y':
    normalize_dataset()
    if os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)
        print("🔄 Progress file reset due to filename reorganization.")

# Get list of images
files = sorted([
    f for f in os.listdir(IMG_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])

if not files:
    print("❌ No image files found in the images folder.")
    exit()

cv2.namedWindow("YOLO Label Tool", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("YOLO Label Tool", mouse_callback)

idx = load_progress()
if idx >= len(files):
    idx = 0
    save_progress(0)

print(f"Starting labeling for {len(files)} images...\n")

while idx < len(files):
    file = files[idx]
    current_image_name = file
    current_image_path = os.path.join(IMG_DIR, file)
    current_label_path = os.path.join(LBL_DIR, os.path.splitext(file)[0] + ".txt")

    # Create empty label file if not exists
    if not os.path.exists(current_label_path):
        print(f"📄 Creating new label file for: {file}")
        open(current_label_path, 'w').close()

    current_img = cv2.imread(current_image_path)
    if current_img is None:
        print(f"❌ Cannot read image: {file}")
        idx += 1
        save_progress(idx)
        continue

    img_h, img_w = current_img.shape[:2]
    current_boxes = load_labels(current_label_path)
    edit_mode = persistent_edit

    print(f"📸 Processing: {current_image_name} ({idx + 1}/{len(files)})")

    while True:
        display = draw()
        cv2.imshow("YOLO Label Tool", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            save_progress(idx)
            print("👋 Exiting program...")
            cv2.destroyAllWindows()
            exit()

        elif key == ord('s'):           # Save and next
            save_labels()
            idx += 1
            save_progress(idx)
            break

        elif key in [ord('n'), ord('0')]:  # Next without saving
            idx += 1
            save_progress(idx)
            break

        elif key == ord('b') and idx > 0:  # Back
            idx -= 1
            save_progress(idx)
            break

        elif key == ord('e'):           # Toggle edit mode
            edit_mode = not edit_mode
            print(f"✏️ Edit Mode: {'ON' if edit_mode else 'OFF'}")

        elif key == ord('p'):           # Toggle persistent edit
            persistent_edit = not persistent_edit
            edit_mode = persistent_edit
            print(f"🔄 Persistent Edit: {'ENABLED' if persistent_edit else 'DISABLED'}")

        elif key == ord('c'):           # Switch class
            current_class = 1 - current_class
            print(f"🔄 Switched to class: {CLASS_NAMES[current_class]}")

        elif key in [ord('d'), ord('z')]:  # Delete last box
            if current_boxes:
                current_boxes.pop()
                print("🗑️ Deleted last box")

        elif key in [ord('+'), ord('=')]:  # Increase thickness
            BOX_THICKNESS = min(10, BOX_THICKNESS + 1)
            print(f"📏 Thickness → {BOX_THICKNESS}")

        elif key in [ord('-'), ord('_')]:  # Decrease thickness
            BOX_THICKNESS = max(1, BOX_THICKNESS - 1)
            print(f"📏 Thickness → {BOX_THICKNESS}")

        elif key == ord('r'):           # Reset image
            current_boxes = load_labels(current_label_path)
            print("🔄 Reset current image")

        elif key == ord('h'):
            print_help()

print("\n🎉 Labeling completed for all images!")
cv2.destroyAllWindows()
