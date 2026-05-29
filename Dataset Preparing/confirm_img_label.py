import cv2
import os

# IMG_DIR = "d:/Machine Vision/practice/soruce/YOLO/dataset/images/train"
# LBL_DIR = "d:/Machine Vision/practice/soruce/YOLO/dataset/labels/train"

#IMG_DIR = "D:\March\MCphase3\chip\dataset_raw\images"
#LBL_DIR = "D:\March\MCphase3\chip\dataset_raw\labels"
IMG_DIR = "D:\March\MCphase3\chip\Preparing_Data\dataset\P\images"
LBL_DIR = "D:\March\MCphase3\chip\Preparing_Data\dataset\P\labels"


PROGRESS_FILE = "progress.txt"

# BOX_W = 40  # adjust based on your island size (pixels)
# BOX_H = 40

current_img = None
current_boxes = []
edit_mode = False
img_w, img_h = 0, 0
current_label_path = ""

drawing = False
start_x, start_y = -1, -1
temp_box = None




# =========================
# PROGRESS CONTROL
# =========================

def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            idx = int(f.read().strip())
            print(f"🔄 Resuming from index: {idx}")
            return idx
    return 0


def save_progress(idx):
    with open(PROGRESS_FILE, "w") as f:
        f.write(str(idx))


# =========================
# LOAD LABELS
# =========================
def load_labels(label_path):
    boxes = []
    if not os.path.exists(label_path):
        return boxes

    with open(label_path, "r") as f:
        for line in f.readlines():
            cls, x, y, bw, bh = map(float, line.strip().split())

            x1 = int((x - bw/2) * img_w)
            y1 = int((y - bh/2) * img_h)
            x2 = int((x + bw/2) * img_w)
            y2 = int((y + bh/2) * img_h)

            boxes.append((x1, y1, x2, y2))

    return boxes

# =========================
# SAVE LABELS
# =========================
def save_labels():
    with open(current_label_path, "w") as f:
        for (x1, y1, x2, y2) in current_boxes:
            w = x2 - x1
            h = y2 - y1

            x_center = (x1 + w/2) / img_w
            y_center = (y1 + h/2) / img_h
            bw = w / img_w
            bh = h / img_h

            f.write(f"0 {x_center} {y_center} {bw} {bh}\n")

    print("✅ Saved:", current_label_path)

# =========================
# DRAW
# =========================
def draw():
    img = current_img.copy()

    # existing boxes
    for (x1, y1, x2, y2) in current_boxes:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # preview box (while dragging)
    if temp_box is not None:
        x1, y1, x2, y2 = temp_box
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    if edit_mode:
        cv2.putText(img, "EDIT MODE", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return img

# =========================
# MOUSE CLICK
# =========================

def remove_nearest_box(x, y):
    global current_boxes

    if len(current_boxes) == 0:
        print("⚠️ No boxes to remove")
        return

    min_dist = float("inf")
    min_idx = -1

    for i, (x1, y1, x2, y2) in enumerate(current_boxes):
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        dist = (cx - x) ** 2 + (cy - y) ** 2  # no sqrt (faster)

        if dist < min_dist:
            min_dist = dist
            min_idx = i

    removed = current_boxes.pop(min_idx)
    print(f"🗑️ Removed nearest box: {removed}")

def mouse_callback(event, x, y, flags, param):
    global drawing, start_x, start_y, temp_box, current_boxes

    if not edit_mode:
        return

    # LEFT BUTTON DOWN → start drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_x, start_y = x, y

    # MOUSE MOVE → update preview
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        temp_box = (start_x, start_y, x, y)

    # LEFT BUTTON UP → finalize box
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

        x1 = min(start_x, x)
        y1 = min(start_y, y)
        x2 = max(start_x, x)
        y2 = max(start_y, y)

        # clamp
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img_w, x2)
        y2 = min(img_h, y2)

        # avoid tiny boxes (noise)
        if (x2 - x1) > 10 and (y2 - y1) > 10:
            current_boxes.append((x1, y1, x2, y2))
            print(f"➕ Box added: {(x1, y1, x2, y2)}")
        else:
            print("⚠️ Box too small, ignored")

        temp_box = None

    # RIGHT CLICK → delete nearest
    elif event == cv2.EVENT_RBUTTONDOWN:
        remove_nearest_box(x, y)

# =========================
# MAIN LOOP
# =========================
cv2.namedWindow("viewer")
cv2.setMouseCallback("viewer", mouse_callback)

files = [f for f in os.listdir(IMG_DIR) if f.endswith((".jpg", ".png"))]

idx = load_progress()

while idx < len(files):
    file = files[idx]

    img_path = os.path.join(IMG_DIR, file)
    current_label_path = os.path.join(LBL_DIR, file.replace(".jpg", ".txt").replace(".png", ".txt"))

    current_img = cv2.imread(img_path)
    img_h, img_w = current_img.shape[:2]

    current_boxes = load_labels(current_label_path)
    edit_mode = False

    while True:
        display = draw()
        cv2.imshow("viewer", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            save_progress(idx)
            exit()

        elif key == ord('0'):  # next image
            idx += 1
            save_progress(idx)
            break

        elif key == ord('e'):  # enter edit mode
            edit_mode = not edit_mode
            print("✏️ Edit mode:", edit_mode)

        elif key == ord('s'):  # save
            save_labels()

        elif key == ord('d'):  # delete last box
            if len(current_boxes) > 0:
                removed = current_boxes.pop()
                print(f"🗑️ Removed box: {removed}")
            else:
                print("⚠️ No box to remove")

        elif key == ord('r'):
            idx = 0
            save_progress(idx)
            print("🔄 Reset to beginning")
            break

if idx == len(files):
    idx = 0
    save_progress(idx)

    
cv2.destroyAllWindows()