#bestVersion
import cv2
import os
import shutil

# ========================= CONFIG =========================
IMG_DIR = r"D:\March\MCphase3\chip\dataset_raw\images"
LBL_DIR = r"D:\March\MCphase3\chip\dataset_raw\labels"

PROGRESS_FILE = "progress.txt"

CLASS_NAMES = {0: "NG", 1: "OK"}
current_class = 0
persistent_edit = False
BOX_THICKNESS = 1

# Global variables
current_img = None
current_boxes = []
edit_mode = False
img_w, img_h = 0, 0
current_label_path = ""
current_image_name = ""

drawing = False
start_x, start_y = -1, -1
temp_box = None

# ========================= UTILITY =========================
def ensure_dirs():
    os.makedirs(IMG_DIR, exist_ok=True)
    os.makedirs(LBL_DIR, exist_ok=True)

def normalize_dataset():
    print("\n🔄 กำลังจัดระเบียบชื่อไฟล์ภาพทั้งหมด...")
    files = sorted([f for f in os.listdir(IMG_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    
    if not files:
        print("ไม่พบไฟล์ภาพ")
        return

    print(f"พบ {len(files)} ไฟล์ภาพ")

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
            print(f"❌ Error temp: {old_name} → {e}")

    for i, (temp_name, ext) in enumerate(temp_list, 1):
        temp_path = os.path.join(IMG_DIR, temp_name)
        new_name = f"img_{i:04d}{ext}"
        new_path = os.path.join(IMG_DIR, new_name)
        
        try:
            shutil.move(temp_path, new_path)
            print(f"   Renamed: {files[i-1]} → {new_name}")
        except Exception as e:
            print(f"❌ Error final: {temp_name} → {e}")
    
    print("✅ จัดระเบียบชื่อไฟล์เสร็จสิ้น\n")

# ========================= PROGRESS =========================
def load_progress():
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, "r") as f:
                return int(f.read().strip())
        except:
            return 0
    return 0

def save_progress(idx):
    with open(PROGRESS_FILE, "w") as f:
        f.write(str(idx))

# ========================= LABELS =========================
def load_labels(label_path):
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path, "r") as f:
        for line in f.readlines():
            if line.strip():
                try:
                    cls, x, y, bw, bh = map(float, line.strip().split())
                    cls = int(cls)
                    # ป้องกัน class ที่ไม่ถูกต้อง
                    if cls not in CLASS_NAMES:
                        print(f"⚠️ พบ class {cls} ที่ไม่ถูกต้องในไฟล์ {os.path.basename(label_path)} → เปลี่ยนเป็น 0 (NG)")
                        cls = 0
                    
                    x1 = int((x - bw/2) * img_w)
                    y1 = int((y - bh/2) * img_h)
                    x2 = int((x + bw/2) * img_w)
                    y2 = int((y + bh/2) * img_h)
                    boxes.append((x1, y1, x2, y2, cls))
                except:
                    continue
    return boxes

def save_labels():
    with open(current_label_path, "w") as f:
        for (x1, y1, x2, y2, cls) in current_boxes:
            w = x2 - x1
            h = y2 - y1
            x_center = (x1 + w/2) / img_w
            y_center = (y1 + h/2) / img_h
            bw = w / img_w
            bh = h / img_h
            f.write(f"{cls} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}\n")
    print(f"✅ Saved → {os.path.basename(current_label_path)}")

# ========================= DRAW =========================
def draw():
    img = current_img.copy()

    for (x1, y1, x2, y2, cls) in current_boxes:
        # ป้องกัน KeyError
        class_name = CLASS_NAMES.get(cls, f"UNK({cls})")
        color = (0, 0, 255) if cls == 0 else (0, 255, 0)
        
        cv2.rectangle(img, (x1, y1), (x2, y2), color, BOX_THICKNESS)
        cv2.putText(img, class_name, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, max(2, BOX_THICKNESS))

    if temp_box:
        x1, y1, x2, y2 = temp_box
        color = (0, 0, 255) if current_class == 0 else (0, 255, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, BOX_THICKNESS)

    mode_text = "PERSISTENT EDIT" if persistent_edit else "EDIT MODE" if edit_mode else "VIEW MODE"
    cv2.putText(img, f"Image: {current_image_name}  |  Class: {CLASS_NAMES[current_class]}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    cv2.putText(img, f"Mode: {mode_text}  |  Boxes: {len(current_boxes)}  |  Thickness: {BOX_THICKNESS}", 
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    cv2.putText(img, "Press H for Help", (10, img_h - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    return img

# ========================= MOUSE & HELP =========================
def remove_nearest_box(x, y):
    global current_boxes
    if not current_boxes: 
        return
    min_idx = min(range(len(current_boxes)), 
                  key=lambda i: ((current_boxes[i][0]+current_boxes[i][2])//2 - x)**2 + 
                               ((current_boxes[i][1]+current_boxes[i][3])//2 - y)**2)
    current_boxes.pop(min_idx)
    print("🗑️ Removed nearest box")

def mouse_callback(event, x, y, flags, param):
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
    print("\n" + "="*70)
    print("                   YOLO LABELING TOOL - HELP")
    print("="*70)
    print("S     = Save + Next")
    print("N / 0 = Next (ไม่เซฟ)")
    print("B     = Back")
    print("E     = Toggle Edit Mode")
    print("P     = Persistent Edit")
    print("C     = Switch Class (NG ↔ OK)")
    print("D / Z = Delete last box")
    print("+ / =  = Increase Thickness")
    print("- / _  = Decrease Thickness")
    print("R     = Reset current image")
    print("H     = Show Help")
    print("Q     = Quit")
    print("Right Click = Delete nearest box")
    print("="*70 + "\n")

# ========================= MAIN PROGRAM =========================
ensure_dirs()
print_help()

reorganize = input("ต้องการจัดระเบียบชื่อไฟล์ทั้งหมดเป็น img_XXXX ก่อนเริ่มไหม? (y/n): ").strip().lower()

if reorganize == 'y':
    normalize_dataset()
    if os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)
        print("🔄 รีเซ็ต progress เนื่องจากจัดระเบียบไฟล์ใหม่")

files = sorted([f for f in os.listdir(IMG_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))])

if not files:
    print("❌ ไม่พบไฟล์ภาพในโฟลเดอร์ images")
    exit()

cv2.namedWindow("YOLO Label Tool", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("YOLO Label Tool", mouse_callback)

idx = load_progress()
if idx >= len(files):
    idx = 0
    save_progress(0)

print(f"เริ่ม Labeling ทั้งหมด {len(files)} ภาพ...\n")

while idx < len(files):
    file = files[idx]
    current_image_name = file
    current_image_path = os.path.join(IMG_DIR, file)
    current_label_path = os.path.join(LBL_DIR, os.path.splitext(file)[0] + ".txt")

    if not os.path.exists(current_label_path):
        print(f"📄 สร้าง label ใหม่สำหรับ: {file}")
        open(current_label_path, 'w').close()

    current_img = cv2.imread(current_image_path)
    if current_img is None:
        print(f"❌ ไม่สามารถอ่านภาพ {file} ได้")
        idx += 1
        save_progress(idx)
        continue

    img_h, img_w = current_img.shape[:2]
    current_boxes = load_labels(current_label_path)
    edit_mode = persistent_edit

    print(f"📸 กำลังทำ: {current_image_name} ({idx+1}/{len(files)})")

    while True:
        display = draw()
        cv2.imshow("YOLO Label Tool", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            save_progress(idx)
            print("👋 ออกจากโปรแกรม")
            cv2.destroyAllWindows()
            exit()

        elif key == ord('s'):
            save_labels()
            idx += 1
            save_progress(idx)
            break

        elif key in [ord('n'), ord('0')]:
            idx += 1
            save_progress(idx)
            break

        elif key == ord('b') and idx > 0:
            idx -= 1
            save_progress(idx)
            break

        elif key == ord('e'):
            edit_mode = not edit_mode
            print(f"✏️ Edit Mode: {'ON' if edit_mode else 'OFF'}")

        elif key == ord('p'):
            persistent_edit = not persistent_edit
            edit_mode = persistent_edit
            print(f"🔄 Persistent Edit: {'ENABLED' if persistent_edit else 'DISABLED'}")

        elif key == ord('c'):
            current_class = 1 - current_class
            print(f"🔄 Class → {CLASS_NAMES[current_class]}")

        elif key in [ord('d'), ord('z')]:
            if current_boxes:
                current_boxes.pop()
                print("🗑️ Deleted last box")

        elif key in [ord('+'), ord('=')]:
            BOX_THICKNESS = min(10, BOX_THICKNESS + 1)
            print(f"📏 Thickness → {BOX_THICKNESS}")

        elif key in [ord('-'), ord('_')]:
            BOX_THICKNESS = max(1, BOX_THICKNESS - 1)
            print(f"📏 Thickness → {BOX_THICKNESS}")

        elif key == ord('r'):
            current_boxes = load_labels(current_label_path)
            print("🔄 Reset current image")

        elif key == ord('h'):
            print_help()

print("\n🎉 เสร็จสิ้นการ Labeling ทั้งหมด!")
cv2.destroyAllWindows()