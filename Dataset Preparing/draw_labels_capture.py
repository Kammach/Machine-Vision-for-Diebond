import cv2
import os
import numpy as np
from picamera2 import Picamera2

# ================== EASY TO EDIT SETTINGS ==================
DATASET_ROOT = "dataset"
IMAGE_DIR = os.path.join(DATASET_ROOT, "images")
LABEL_DIR = os.path.join(DATASET_ROOT, "labels")

INITIAL_BOX_WIDTH = 300
INITIAL_BOX_HEIGHT = 300
MIN_BOX_SIZE = 20
STEP_SIZE = 20

PREVIEW_WIDTH = 2028
PREVIEW_HEIGHT = 1520

HEADER_H = 180
FOOTER_H = 32

FILE_INDEX_WIDTH = 4

CLASS_NAMES = {
    0: "NG",
    1: "OK"
}

CLASS_COLORS = {
    0: (0, 0, 255),     # Red
    1: (0, 255, 0)      # Green
}
# ===========================================================


# ================== CREATE FOLDERS ==================
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(LABEL_DIR, exist_ok=True)


def txt(canvas, text, pos, color, scale=0.68, thickness=1):
    cv2.putText(
        canvas,
        text,
        pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA
    )


def get_folder_counts():
    img_count = len([
        f for f in os.listdir(IMAGE_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    lbl_count = len([
        f for f in os.listdir(LABEL_DIR)
        if f.lower().endswith(".txt")
    ])
    return img_count, lbl_count


def get_next_index():
    max_index = -1

    for f in os.listdir(IMAGE_DIR):
        stem, ext = os.path.splitext(f)
        if ext.lower() not in (".jpg", ".jpeg", ".png"):
            continue

        if stem.startswith("img_"):
            number_part = stem[4:]
            if number_part.isdigit():
                idx = int(number_part)
                if idx > max_index:
                    max_index = idx

    return max_index + 1


def make_filename(index, ext=".jpg"):
    stem = f"img_{index:0{FILE_INDEX_WIDTH}d}"
    return stem + ext, stem


def clamp_box(x_center, y_center, box_w, box_h, frame_w, frame_h):
    box_w = min(box_w, frame_w)
    box_h = min(box_h, frame_h)

    x1 = int(x_center - box_w // 2)
    y1 = int(y_center - box_h // 2)

    x1 = max(0, min(x1, frame_w - box_w))
    y1 = max(0, min(y1, frame_h - box_h))

    x2 = x1 + box_w
    y2 = y1 + box_h

    return x1, y1, x2, y2


def box_to_yolo(box, frame_w, frame_h):
    x1, y1, x2, y2, cls_id = box

    bw = x2 - x1
    bh = y2 - y1
    x_center = x1 + bw / 2.0
    y_center = y1 + bh / 2.0

    return (
        cls_id,
        x_center / frame_w,
        y_center / frame_h,
        bw / frame_w,
        bh / frame_h,
    )


def draw_text_lines(img, lines, x=10, y=30, line_gap=24, scale=0.6, color=(255, 255, 255), thickness=1):
    for i, line in enumerate(lines):
        cv2.putText(
            img,
            line,
            (x, y + i * line_gap),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
            cv2.LINE_AA
        )


def draw_box(img, box, color, thickness=2, y_offset=0):
    x1, y1, x2, y2, cls_id = box
    class_name = CLASS_NAMES.get(cls_id, f"UNK({cls_id})")

    y1 += y_offset
    y2 += y_offset

    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    cv2.putText(
        img,
        class_name,
        (x1, max(20, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        max(1, thickness),
        cv2.LINE_AA
    )


def save_sample(frame_bgr, boxes, index):
    file_name, stem = make_filename(index, ".jpg")
    img_path = os.path.join(IMAGE_DIR, file_name)
    lbl_path = os.path.join(LABEL_DIR, stem + ".txt")

    frame_h, frame_w = frame_bgr.shape[:2]

    ok = cv2.imwrite(img_path, frame_bgr)
    if not ok:
        raise RuntimeError(f"Failed to save image: {img_path}")

    with open(lbl_path, "w", encoding="utf-8") as f:
        for box in boxes:
            cls_id, x_center, y_center, bw, bh = box_to_yolo(box, frame_w, frame_h)
            f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}\n")

    return img_path, lbl_path


def print_help():
    print("\n" + "=" * 80)
    print("YOLO DATASET COLLECTOR - HELP")
    print("=" * 80)
    print("f       = Freeze current frame")
    print("u       = Unfreeze (discard current temporary boxes)")
    print("s       = Save current frozen frame + labels")
    print("m       = Switch class (NG <-> OK)")
    print("z       = Delete last box")
    print("c       = Clear all boxes")
    print("+ / =   = Increase box width and height")
    print("-       = Decrease box width and height")
    print("[       = Increase width only")
    print("]       = Decrease width only")
    print(";       = Increase height only")
    print("'       = Decrease height only")
    print("r       = Reset box size to default")
    print("q / ESC = Quit")
    print("Mouse left click inside image area when frozen = Add one label box")
    print("=" * 80 + "\n")


class State:
    def __init__(self):
        self.selected_class = 0
        self.freeze_mode = False
        self.frozen_frame = None
        self.current_boxes = []
        self.mouse_x = PREVIEW_WIDTH // 2
        self.mouse_y = HEADER_H + PREVIEW_HEIGHT // 2
        self.box_width = INITIAL_BOX_WIDTH
        self.box_height = INITIAL_BOX_HEIGHT
        self.next_index = get_next_index()
        self.last_saved = "-"
        self.status = "Ready"

    def active_folder_name(self):
        return DATASET_ROOT

    def current_class_name(self):
        return CLASS_NAMES[self.selected_class]

    def current_folder_count(self):
        img_count, _ = get_folder_counts()
        return img_count


st = State()


def draw_header(canvas, st):
    h, w = canvas.shape[:2]

    cv2.rectangle(canvas, (0, 0), (w, HEADER_H), (18, 18, 18), -1)
    cv2.line(canvas, (0, HEADER_H - 1), (w, HEADER_H - 1), (70, 70, 70), 2)

    left_x = 24
    mid1_x = int(w * 0.28)
    mid2_x = int(w * 0.53)
    right_x = int(w * 0.78)

    cv2.line(canvas, (int(w * 0.24), 14), (int(w * 0.24), HEADER_H - 40), (55, 55, 55), 1)
    cv2.line(canvas, (int(w * 0.49), 14), (int(w * 0.49), HEADER_H - 40), (55, 55, 55), 1)
    cv2.line(canvas, (int(w * 0.74), 14), (int(w * 0.74), HEADER_H - 40), (55, 55, 55), 1)

    title_color = (170, 170, 170)
    txt(canvas, "SYSTEM",   (left_x, 28),  title_color, 0.95, 2)
    txt(canvas, "DATASET",  (mid1_x, 28),  title_color, 0.95, 2)
    txt(canvas, "SESSION",  (mid2_x, 28),  title_color, 0.95, 2)
    txt(canvas, "CONTROLS", (right_x, 28), title_color, 0.95, 2)

    mode = "FREEZE" if st.freeze_mode else "LIVE"

    txt(canvas, f"MODE: {mode}", (left_x, 70), (0, 255, 255), 1.15, 2)
    txt(canvas, f"CLASS: {st.current_class_name()}", (left_x, 112), CLASS_COLORS[st.selected_class], 1.05, 2)

    txt(canvas, f"OUTPUT: {DATASET_ROOT}", (mid1_x, 70), (255, 255, 255), 0.92, 2)
    txt(canvas, f"SAVED: {st.current_folder_count()}", (mid1_x, 112), (255, 255, 255), 1.00, 2)

    txt(canvas, f"NEXT: img_{st.next_index:0{FILE_INDEX_WIDTH}d}", (mid2_x, 70), (0, 255, 255), 0.95, 2)
    txt(canvas, f"LAST: {st.last_saved}", (mid2_x, 112), (255, 255, 255), 0.82, 2)

    txt(canvas, "F Freeze    S Save    M Class", (right_x, 66), (255, 255, 255), 0.80, 2)
    txt(canvas, "Z Undo      C Clear    Q Quit", (right_x, 94), (255, 255, 255), 0.80, 2)

    cv2.rectangle(canvas, (0, HEADER_H - 32), (w, HEADER_H), (30, 30, 30), -1)
    txt(canvas, f"STATUS: {st.status}", (24, HEADER_H - 9), (0, 255, 255), 0.82, 2)
    txt(canvas, "+/= Grow    - Shrink   [ ] Width   ; ' Height", (mid2_x, HEADER_H - 9), (255, 255, 255), 0.82, 2)


def draw_footer(canvas, st, frame_h):
    y0 = HEADER_H + PREVIEW_HEIGHT
    cv2.rectangle(canvas, (0, y0), (canvas.shape[1], y0 + FOOTER_H), (18, 18, 18), -1)
    cv2.line(canvas, (0, y0), (canvas.shape[1], y0), (70, 70, 70), 1)

    footer_text = f"Boxes: {len(st.current_boxes)} | Preview Box: {st.box_width} x {st.box_height} | Current Class: {st.current_class_name()}"
    txt(canvas, footer_text, (18, y0 + 22), (220, 220, 220), 0.55, 1)


# ================== CAMERA INIT ==================
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (PREVIEW_WIDTH, PREVIEW_HEIGHT)})
picam2.configure(config)
picam2.start()

print(f"Stable preview/image area: {PREVIEW_WIDTH} x {PREVIEW_HEIGHT}")

print_help()
print(f"Output folder: {DATASET_ROOT}")
print(f"Next file index: {st.next_index:0{FILE_INDEX_WIDTH}d}")
print("Ready.\n")


def mouse_callback(event, x, y, flags, param):
    st.mouse_x = x
    st.mouse_y = y

    image_y = y - HEADER_H

    if event == cv2.EVENT_LBUTTONDOWN and st.freeze_mode and st.frozen_frame is not None:
        if not (0 <= x < PREVIEW_WIDTH and 0 <= image_y < PREVIEW_HEIGHT):
            return

        frame_h, frame_w = st.frozen_frame.shape[:2]

        bx = min(st.box_width, frame_w)
        by = min(st.box_height, frame_h)

        x1, y1, x2, y2 = clamp_box(x, image_y, bx, by, frame_w, frame_h)
        st.current_boxes.append((x1, y1, x2, y2, st.selected_class))

        st.status = f"Added label: {CLASS_NAMES[st.selected_class]} ({len(st.current_boxes)} total)"
        print(
            f"Added label | Class: {CLASS_NAMES[st.selected_class]} "
            f"| Box: ({x1}, {y1}) -> ({x2}, {y2}) "
            f"| Size: {x2 - x1}x{y2 - y1} "
            f"| Total: {len(st.current_boxes)}"
        )


WINDOW_H = HEADER_H + PREVIEW_HEIGHT + FOOTER_H

cv2.namedWindow("YOLO Dataset Collector", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("YOLO Dataset Collector", mouse_callback)
cv2.resizeWindow("YOLO Dataset Collector", PREVIEW_WIDTH, WINDOW_H)


while True:
    if st.freeze_mode and st.frozen_frame is not None:
        image_frame = st.frozen_frame.copy()
    else:
        rgb = picam2.capture_array()
        image_frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    frame_h, frame_w = image_frame.shape[:2]
    ok_count, ng_count = get_folder_counts()

    current_color = CLASS_COLORS[st.selected_class]

    if st.freeze_mode:
        for box in st.current_boxes:
            cls_id = box[4]
            color = CLASS_COLORS.get(cls_id, (255, 255, 255))
            draw_box(image_frame, box, color, thickness=2, y_offset=0)

        mx = max(0, min(st.mouse_x, frame_w - 1))
        my = max(0, min(st.mouse_y - HEADER_H, frame_h - 1))

        preview_w = min(st.box_width, frame_w)
        preview_h = min(st.box_height, frame_h)
        px1, py1, px2, py2 = clamp_box(mx, my, preview_w, preview_h, frame_w, frame_h)

        cv2.rectangle(image_frame, (px1, py1), (px2, py2), current_color, 2)
        cv2.circle(image_frame, (mx, my), 4, current_color, -1)

    canvas = np.zeros((WINDOW_H, PREVIEW_WIDTH, 3), dtype=np.uint8)

    canvas[HEADER_H:HEADER_H + PREVIEW_HEIGHT, 0:PREVIEW_WIDTH] = image_frame

    draw_header(canvas, st)
    draw_footer(canvas, st, frame_h)

    cv2.imshow("YOLO Dataset Collector", canvas)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q') or key == 27:
        st.status = "Exiting"
        print("Exiting program.")
        break

    elif key == ord('f') and not st.freeze_mode:
        st.freeze_mode = True
        st.frozen_frame = image_frame.copy()
        st.current_boxes = []
        st.status = "Frozen - click inside image area to add labels"
        print("Frame frozen. Click on the image to add labels.")

    elif key == ord('u') and st.freeze_mode:
        st.freeze_mode = False
        st.frozen_frame = None
        st.current_boxes = []
        st.status = "Unfrozen - temporary labels cleared"
        print("Unfrozen. Temporary labels cleared.")

    elif key == ord('s') and st.freeze_mode and st.frozen_frame is not None:
        try:
            if len(st.current_boxes) == 0:
                st.status = "Saving image with empty label file"
                print("Saved image with empty label file.")
            else:
                st.status = f"Saving {len(st.current_boxes)} labels"
                print(f"Saving {len(st.current_boxes)} labels...")

            img_path, lbl_path = save_sample(st.frozen_frame, st.current_boxes, st.next_index)
            st.last_saved = os.path.basename(img_path)
            st.status = f"Saved: {st.last_saved}"

            print(f"Saved image: {img_path}")
            print(f"Saved label: {lbl_path}")

            st.next_index += 1
            st.freeze_mode = False
            st.frozen_frame = None
            st.current_boxes = []
        except Exception as e:
            st.status = f"Save failed: {e}"
            print(f"[ERROR] {e}")

    elif key == ord('m'):
        st.selected_class = 1 - st.selected_class
        st.status = f"Current class changed to {CLASS_NAMES[st.selected_class]}"
        print(f"Current class changed to: {CLASS_NAMES[st.selected_class]}")

    elif key == ord('z') and st.freeze_mode and len(st.current_boxes) > 0:
        st.current_boxes.pop()
        st.status = f"Deleted last box. Remaining: {len(st.current_boxes)}"
        print(f"Deleted last box. Remaining: {len(st.current_boxes)}")

    elif key == ord('c') and st.freeze_mode:
        st.current_boxes = []
        st.status = "Cleared all boxes"
        print("Cleared all boxes on current frozen frame.")

    elif key in [ord('+'), ord('=')]:
        st.box_width += STEP_SIZE
        st.box_height += STEP_SIZE
        st.status = f"Box size increased to {st.box_width} x {st.box_height}"
        print(f"Box size increased -> {st.box_width} x {st.box_height}")

    elif key == ord('-'):
        st.box_width = max(MIN_BOX_SIZE, st.box_width - STEP_SIZE)
        st.box_height = max(MIN_BOX_SIZE, st.box_height - STEP_SIZE)
        st.status = f"Box size decreased to {st.box_width} x {st.box_height}"
        print(f"Box size decreased -> {st.box_width} x {st.box_height}")

    elif key == ord('['):
        st.box_width = max(MIN_BOX_SIZE, st.box_width - STEP_SIZE)
        st.status = f"Width decreased to {st.box_width} x {st.box_height}"
        print(f"Width decreased -> {st.box_width} x {st.box_height}")

    elif key == ord(']'):
        st.box_width += STEP_SIZE
        st.status = f"Width increased to {st.box_width} x {st.box_height}"
        print(f"Width increased -> {st.box_width} x {st.box_height}")

    elif key == ord(';'):
        st.box_height += STEP_SIZE
        st.status = f"Height increased to {st.box_width} x {st.box_height}"
        print(f"Height increased -> {st.box_width} x {st.box_height}")

    elif key == ord("'"):
        st.box_height = max(MIN_BOX_SIZE, st.box_height - STEP_SIZE)
        st.status = f"Height decreased to {st.box_width} x {st.box_height}"
        print(f"Height decreased -> {st.box_width} x {st.box_height}")

    elif key == ord('r'):
        st.box_width = INITIAL_BOX_WIDTH
        st.box_height = INITIAL_BOX_HEIGHT
        st.status = f"Box size reset to {st.box_width} x {st.box_height}"
        print(f"Box size reset to {st.box_width} x {st.box_height}")

# Cleanup
picam2.stop()
cv2.destroyAllWindows()
print("Program closed.")
