import os
import cv2
import gi
import time
import queue
import threading
import numpy as np
import hailo

from pathlib import Path

os.environ["GST_PLUGIN_FEATURE_RANK"] = "vaapidecodebin:NONE"

gi.require_version("Gst", "1.0")
from gi.repository import Gst

from hailo_apps.python.pipeline_apps.detection.detection_pipeline import GStreamerDetectionApp
from hailo_apps.python.core.common.buffer_utils import get_caps_from_pad, get_numpy_from_buffer
from hailo_apps.python.core.gstreamer.gstreamer_app import app_callback_class


# =========================================================
# CONFIG
# =========================================================

CFG = {
    "WIN": "Chip Dataset Collector",
    "DATASET_DIR": "/home/pi/hailo-apps/hailo_apps/python/mcphase3/dataset",
    "CONF_THRES": 0.20,
    "HEADER_H": 96,
    "FONT": cv2.FONT_HERSHEY_TRIPLEX,

    "CLASSES": [
        "OK",   # Chip
        "NG",   # No chip
        "GL",   # Glue
        "MA",   # Misalig
        "DF",   # defect
        "OG",   # Glue_overflow
        "class6",
    ],

    "FREEZE_FOLDER": "P",

    "KEY_SAVE": ord("s"),
    "KEY_FREEZE": ord("f"),
    "KEY_UNDO": ord("z"),
    "KEY_NEXT_CLASS": ord("d"),
    "KEY_PREV_CLASS": ord("a"),
    "KEY_OVERLAY_MODE": ord("o"),
    "KEY_QUIT": ord("q"),

    # Overlay fine control
    "OVERLAY_STEP": 1,
    "OVERLAY_MIN_SIZE": 4,

    # 1 = left expand
    "KEY_LEFT_EXPAND": ord("1"),
    "KEY_LEFT_SHRINK": ord("2"),
    "KEY_RIGHT_EXPAND": ord("3"),
    "KEY_RIGHT_SHRINK": ord("4"),
    "KEY_TOP_EXPAND": ord("5"),
    "KEY_TOP_SHRINK": ord("6"),
    "KEY_BOTTOM_EXPAND": ord("7"),
    "KEY_BOTTOM_SHRINK": ord("8"),

    "JPEG_QUALITY": 100,
    "DRAW_BOX_THICKNESS": 2,
    "DRAW_TEXT_THICKNESS": 1,
}

# =========================================================
# FOLDERS
# =========================================================

def ensure_dataset_dirs():
    base = Path(CFG["DATASET_DIR"])
    for name in CFG["CLASSES"] + [CFG["FREEZE_FOLDER"]]:
        (base / name / "images").mkdir(parents=True, exist_ok=True)
        (base / name / "labels").mkdir(parents=True, exist_ok=True)

ensure_dataset_dirs()

# =========================================================
# COLORS
# =========================================================

def build_colors(n: int):
    colors = []
    for i in range(n):
        h = int(180 * i / max(1, n))
        hsv = np.uint8([[[h, 220, 255]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
        colors.append((int(bgr[0]), int(bgr[1]), int(bgr[2])))
    return colors

CLASS_COLORS = build_colors(len(CFG["CLASSES"]))

# =========================================================
# SMALL UTILS
# =========================================================

def txt(img, text, pos, color=(255, 255, 255), scale=0.7, thick=2):
    cv2.putText(img, text, pos, CFG["FONT"], scale, color, thick, cv2.LINE_AA)

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def deep_copy_boxes(boxes):
    return [dict(b) for b in boxes]

def folder_paths(folder_name: str):
    base = Path(CFG["DATASET_DIR"]) / folder_name
    return base / "images", base / "labels"

def count_images(folder_name: str) -> int:
    img_dir, _ = folder_paths(folder_name)
    return len(list(img_dir.glob("img_*.jpg")))

def next_index(folder_name: str) -> int:
    img_dir, _ = folder_paths(folder_name)
    mx = -1
    for f in img_dir.glob("img_*.jpg"):
        try:
            idx = int(f.stem.split("_")[1])
            if idx > mx:
                mx = idx
        except Exception:
            pass
    return mx + 1

def base_filename(folder_name: str) -> str:
    return f"img_{next_index(folder_name):05d}"

def yolo_label_line(class_id, x1, y1, x2, y2, w, h):
    cx = ((x1 + x2) / 2.0) / w
    cy = ((y1 + y2) / 2.0) / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return f"{class_id} {cx:.15f} {cy:.15f} {bw:.15f} {bh:.15f}"

def bbox_contains(b, x, y):
    return b["x1"] <= x <= b["x2"] and b["y1"] <= y <= b["y2"]

def sanitize_boxes_for_save(boxes, w, h):
    out = []
    for b in boxes:
        x1 = clamp(int(b["x1"]), 0, w - 1)
        y1 = clamp(int(b["y1"]), 0, h - 1)
        x2 = clamp(int(b["x2"]), 0, w - 1)
        y2 = clamp(int(b["y2"]), 0, h - 1)

        if x2 <= x1 or y2 <= y1:
            continue

        out.append({
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "class_id": int(b["class_id"]),
        })
    return out

def rebuild_overlay_boxes(base_boxes, pad_left, pad_right, pad_top, pad_bottom, w, h, class_ids=None):
    """
    สร้างกรอบใหม่โดยยึด center เดิมของกรอบโมเดล
    ปรับ 4 ทิศแบบอิสระ:
      - left expand/shrink
      - right expand/shrink
      - top expand/shrink
      - bottom expand/shrink
    """
    out = []

    for i, b in enumerate(base_boxes):
        cls = int(class_ids[i]) if class_ids is not None and i < len(class_ids) else int(b["class_id"])

        x1, y1, x2, y2 = int(b["x1"]), int(b["y1"]), int(b["x2"]), int(b["y2"])
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        left = (cx - x1) + pad_left
        right = (x2 - cx) + pad_right
        top = (cy - y1) + pad_top
        bottom = (y2 - cy) + pad_bottom

        left = max(CFG["OVERLAY_MIN_SIZE"] / 2.0, left)
        right = max(CFG["OVERLAY_MIN_SIZE"] / 2.0, right)
        top = max(CFG["OVERLAY_MIN_SIZE"] / 2.0, top)
        bottom = max(CFG["OVERLAY_MIN_SIZE"] / 2.0, bottom)

        nx1 = clamp(int(round(cx - left)), 0, w - 1)
        nx2 = clamp(int(round(cx + right)), 0, w - 1)
        ny1 = clamp(int(round(cy - top)), 0, h - 1)
        ny2 = clamp(int(round(cy + bottom)), 0, h - 1)

        if nx2 <= nx1:
            if nx1 < w - 1:
                nx2 = nx1 + 1
            else:
                continue

        if ny2 <= ny1:
            if ny1 < h - 1:
                ny2 = ny1 + 1
            else:
                continue

        out.append({
            "x1": nx1,
            "y1": ny1,
            "x2": nx2,
            "y2": ny2,
            "class_id": cls,
        })

    return out

# =========================================================
# STATE
# =========================================================

class State(app_callback_class):
    def __init__(self):
        super().__init__()
        self.q = queue.Queue(2)
        self.lock = threading.Lock()

        self.latest_frame = None
        self.latest_clean_frame = None

        self.live_boxes = []

        self.freeze_mode = False
        self.freeze_frame = None
        self.freeze_clean_frame = None
        self.freeze_base_boxes = []   # กรอบจากโมเดลตอน freeze
        self.freeze_boxes = []        # กรอบที่แก้แล้ว/แสดงจริงตอน freeze

        self.overlay_mode = False
        self.overlay_pad_left = 0
        self.overlay_pad_right = 0
        self.overlay_pad_top = 0
        self.overlay_pad_bottom = 0

        self.overlay_boxes = []

        self.selected_class = 0

        self.undo_stack = []
        self.status = "READY"
        self.last_saved = "-"
        self.current_target_folder = CFG["CLASSES"][0]
        self.current_folder_count = count_images(self.current_target_folder)

        self.fps = 0.0
        self._fps_cnt = 0
        self._fps_t0 = time.time()

    def current_class_name(self):
        return CFG["CLASSES"][self.selected_class]

    def active_folder_name(self):
        return CFG["FREEZE_FOLDER"] if self.freeze_mode else self.current_class_name()

    def active_boxes(self):
        if self.freeze_mode:
            return self.freeze_boxes
        return self.overlay_boxes if self.overlay_mode else self.live_boxes

    def active_clean_frame(self):
        return self.freeze_clean_frame if self.freeze_mode else self.latest_clean_frame

    def refresh_folder_count(self):
        self.current_target_folder = self.active_folder_name()
        self.current_folder_count = count_images(self.current_target_folder)

def sync_overlay_boxes(st: State):
    if not st.overlay_mode:
        return

    if not st.freeze_mode:
        if st.latest_clean_frame is None:
            return
        h, w = st.latest_clean_frame.shape[:2]
        st.overlay_boxes = rebuild_overlay_boxes(
            st.live_boxes,
            st.overlay_pad_left,
            st.overlay_pad_right,
            st.overlay_pad_top,
            st.overlay_pad_bottom,
            w, h
        )
        return

    if st.freeze_clean_frame is None or not st.freeze_base_boxes:
        return

    h, w = st.freeze_clean_frame.shape[:2]
    current_class_ids = [b["class_id"] for b in st.freeze_boxes] if st.freeze_boxes else [b["class_id"] for b in st.freeze_base_boxes]
    st.freeze_boxes = rebuild_overlay_boxes(
        st.freeze_base_boxes,
        st.overlay_pad_left,
        st.overlay_pad_right,
        st.overlay_pad_top,
        st.overlay_pad_bottom,
        w, h,
        class_ids=current_class_ids
    )

# =========================================================
# DRAWING
# =========================================================

def draw_box(frame, b):
    x1, y1, x2, y2 = b["x1"], b["y1"], b["x2"], b["y2"]
    class_id = int(b["class_id"])
    color = CLASS_COLORS[class_id % len(CLASS_COLORS)]

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, CFG["DRAW_BOX_THICKNESS"])

    label = CFG["CLASSES"][class_id]
    text = f"{label}"

    (tw, th), _ = cv2.getTextSize(text, CFG["FONT"], 0.7, CFG["DRAW_TEXT_THICKNESS"])
    y = y1 - 12 if y1 > 40 else y2 + 28

    top = y - th - 10
    bottom = y + 6
    if top < 0:
        top = 0
        bottom = th + 16

    cv2.rectangle(frame, (x1, top), (x1 + tw + 14, bottom), color, -1)
    txt(frame, text, (x1 + 7, y), (0, 0, 0), 0.7, CFG["DRAW_TEXT_THICKNESS"])

def draw_header(canvas, st: State):
    h, w = canvas.shape[:2]

    cv2.rectangle(canvas, (0, 0), (w, CFG["HEADER_H"]), (10, 10, 10), -1)
    cv2.line(canvas, (0, CFG["HEADER_H"] - 1), (w, CFG["HEADER_H"] - 1), (60, 60, 60), 1)

    mode = "FREEZE" if st.freeze_mode else "LIVE"
    if st.overlay_mode:
        mode += " + OVERLAY"

    folder = st.active_folder_name()

    txt(canvas, f"MODE: {mode}", (18, 30), (0, 255, 255), 0.68, 1)
    txt(canvas, f"CLASS: {st.current_class_name()}", (18, 64), CLASS_COLORS[st.selected_class], 0.68, 1)
    txt(canvas, f"FOLDER: {folder}", (300, 30), (255, 255, 255), 0.68, 1)
    txt(canvas, f"SAVED: {st.current_folder_count}", (300, 64), (255, 255, 255), 0.68, 1)
    txt(canvas, f"LAST: {st.last_saved}", (475, 30), (0, 255, 255), 0.68, 1)
    txt(canvas, f"STATUS: {st.status}", (475, 90), (0, 255, 255), 0.5, 1)

    #txt(canvas, f"L:{st.overlay_pad_left:+d}  R:{st.overlay_pad_right:+d}  T:{st.overlay_pad_top:+d}  B:{st.overlay_pad_bottom:+d}",
    #    (750, 30), (255, 200, 80), 0.62, 1)

    help1 = "F=Freeze  S=Save  O=Overlay  A/D=Class  Z=Undo  Q=Quit"
    help2 = "1/2--3/4--5/6--7/8"
    txt(canvas, help1, (18, 88), (255, 255, 255), 0.4, 1)
    txt(canvas, help2, (475, 60), (255, 255, 255), 0.4, 1)


"""
def draw_header(canvas, st: State):
    h, w = canvas.shape[:2]
    header_h = 100
    # พื้นหลังหลัก (Dark Modern)
    cv2.rectangle(canvas, (0, 0), (w, header_h), (0, 0, 0), -1)
    
    # Gradient accent บนสุด (บางๆ)
    cv2.rectangle(canvas, (0, 0), (w, 4), (0, 0, 0), -1)

    # Divider บรรทัดล่าง
    cv2.line(canvas, (0, header_h - 1), (w, header_h - 1), (45, 45, 50), 1)

    # ==================== LEFT SECTION ====================
    # MODE
    mode = "FREEZE" if st.freeze_mode else "LIVE"
    mode_color = (0, 240, 120) if not st.freeze_mode else (255, 80, 80)
    
    txt(canvas, "MODE", (20, 32), (140, 140, 150), 0.65, 1)
    txt(canvas, mode, (20, 62), mode_color, 1.05, 2)

    # CLASS
    class_name = st.current_class_name()
    class_color = CLASS_COLORS[st.selected_class]
    
    txt(canvas, "CLASS", (180, 32), (140, 140, 150), 0.65, 1)
    txt(canvas, class_name, (180, 62), class_color, 1.0, 2)

    # ==================== CENTER SECTION ====================
    folder = st.active_folder_name()
    
    txt(canvas, "FOLDER", (420, 28), (140, 140, 150), 0.62, 1)
    txt(canvas, folder[:38], (420, 55), (255, 255, 255), 0.82, 2)   # จำกัดความยาว

    # Stats
    txt(canvas, f"SAVED", (720, 28), (140, 140, 150), 0.62, 1)
    txt(canvas, str(st.current_folder_count), (720, 55), (0, 255, 180), 0.95, 2)

    txt(canvas, "LAST", (850, 28), (140, 140, 150), 0.62, 1)
    txt(canvas, st.last_saved, (850, 55), (220, 220, 220), 0.78, 2)

    # ==================== RIGHT SECTION ====================
    # Status
    status_color = (0, 255, 140) if "OK" in st.status or "Saved" in st.status else (255, 200, 80)
    txt(canvas, "STATUS", (1070, 28), (140, 140, 150), 0.62, 1)
    txt(canvas, st.status, (1070, 70), status_color, 0.85, 2)

    # Help
    help_text = "F=Freeze  S=Save  A/D=Class  Z=Undo  Q=Quit"
    txt(canvas, help_text, (20, 88), (110, 110, 120), 0.58, 1)

    # Divider แยกโซน
    cv2.line(canvas, (395, 12), (395, 88), (45, 45, 50), 1)
    cv2.line(canvas, (695, 12), (695, 88), (45, 45, 50), 1)
    cv2.line(canvas, (1045, 12), (1045, 88), (45, 45, 50), 1)
"""


def draw_frame(st: State, raw_frame):
    canvas = raw_frame.copy()
    for b in st.active_boxes():
        draw_box(canvas, b)

    out = np.zeros((canvas.shape[0] + CFG["HEADER_H"], canvas.shape[1], 3), dtype=np.uint8)
    out[CFG["HEADER_H"]:] = canvas
    draw_header(out, st)
    return out

# =========================================================
# SAVE AND UNDO
# =========================================================

def save_current(st: State):
    folder_name = st.active_folder_name()
    img_dir, lbl_dir = folder_paths(folder_name)

    frame = st.active_clean_frame()
    boxes = st.active_boxes()

    if frame is None:
        st.status = "NO FRAME"
        return

    h, w = frame.shape[:2]
    base = base_filename(folder_name)
    img_path = img_dir / f"{base}.jpg"
    lbl_path = lbl_dir / f"{base}.txt"

    if st.freeze_mode:
        boxes_to_save = deep_copy_boxes(boxes)
    else:
        boxes_to_save = deep_copy_boxes(boxes)
        for b in boxes_to_save:
            b["class_id"] = st.selected_class

    boxes_to_save = sanitize_boxes_for_save(boxes_to_save, w, h)

    try:
        cv2.imwrite(str(img_path), frame, [cv2.IMWRITE_JPEG_QUALITY, CFG["JPEG_QUALITY"]])
        with open(lbl_path, "w", encoding="utf-8") as f:
            for b in boxes_to_save:
                f.write(
                    yolo_label_line(
                        b["class_id"],
                        b["x1"], b["y1"], b["x2"], b["y2"],
                        w, h
                    ) + "\n"
                )

        st.undo_stack.append((str(img_path), str(lbl_path)))
        st.last_saved = base
        st.status = f"SAVED {base} to {folder_name}"
        st.refresh_folder_count()

    except Exception as e:
        st.status = f"{e}"

def undo_last(st: State):
    if not st.undo_stack:
        st.status = "UNDO EMPTY"
        return

    img_path, lbl_path = st.undo_stack.pop()

    try:
        if os.path.exists(img_path):
            os.remove(img_path)
        if os.path.exists(lbl_path):
            os.remove(lbl_path)

        st.status = f"UNDO OK UNSAVED {len(st.undo_stack)} PIC"
        st.last_saved = Path(img_path).stem if st.undo_stack else "-"
        st.refresh_folder_count()

    except Exception as e:
        st.status = f"UNDO ERROR {e}"

# =========================================================
# MOUSE
# =========================================================

def on_mouse(event, x, y, flags, st: State):
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    if not st.freeze_mode:
        return

    y -= CFG["HEADER_H"]
    if y < 0:
        return

    with st.lock:
        boxes = st.freeze_boxes

        for idx, b in enumerate(boxes):
            if bbox_contains(b, x, y):
                b["class_id"] = (int(b["class_id"]) + 1) % len(CFG["CLASSES"])
                st.status = f"BOX {idx} -> {CFG['CLASSES'][b['class_id']]}"
                if st.overlay_mode:
                    sync_overlay_boxes(st)
                break

# =========================================================
# DISPLAY THREAD
# =========================================================

def adjust_overlay(st: State, side: str, delta: int):
    if side == "left":
        st.overlay_pad_left += delta
    elif side == "right":
        st.overlay_pad_right += delta
    elif side == "top":
        st.overlay_pad_top += delta
    elif side == "bottom":
        st.overlay_pad_bottom += delta

    if st.overlay_mode:
        sync_overlay_boxes(st)

    st.status = (
        f"OVERLAY L:{st.overlay_pad_left:+d} "
        f"R:{st.overlay_pad_right:+d} "
        f"T:{st.overlay_pad_top:+d} "
        f"B:{st.overlay_pad_bottom:+d}"
    )
    st.refresh_folder_count()

def display_thread(st: State):
    cv2.namedWindow(CFG["WIN"], cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(CFG["WIN"], on_mouse, st)

    while True:
        try:
            raw_frame = st.q.get(timeout=0.05)
        except queue.Empty:
            continue

        st._fps_cnt += 1
        if time.time() - st._fps_t0 >= 1.0:
            st.fps = st._fps_cnt / (time.time() - st._fps_t0)
            st._fps_cnt = 0
            st._fps_t0 = time.time()

        ui = draw_frame(st, raw_frame)
        cv2.imshow(CFG["WIN"], ui)

        key = cv2.waitKey(1) & 0xFF

        if key == CFG["KEY_QUIT"]:
            break

        elif key == CFG["KEY_FREEZE"]:
            with st.lock:
                if not st.freeze_mode:
                    if st.latest_clean_frame is not None:
                        st.freeze_mode = True
                        st.freeze_frame = st.latest_frame.copy() if st.latest_frame is not None else None
                        st.freeze_clean_frame = st.latest_clean_frame.copy()

                        st.freeze_base_boxes = deep_copy_boxes(st.live_boxes)
                        st.freeze_boxes = deep_copy_boxes(st.live_boxes)

                        if st.overlay_mode:
                            sync_overlay_boxes(st)

                        st.status = "FREEZE MODE"
                        st.refresh_folder_count()
                else:
                    st.freeze_mode = False
                    st.freeze_frame = None
                    st.freeze_clean_frame = None
                    st.freeze_base_boxes = []
                    st.freeze_boxes = []
                    st.status = "LIVE MODE"
                    st.refresh_folder_count()

        elif key == CFG["KEY_SAVE"]:
            with st.lock:
                save_current(st)

        elif key == CFG["KEY_UNDO"]:
            with st.lock:
                undo_last(st)

        elif key == CFG["KEY_NEXT_CLASS"]:
            with st.lock:
                st.selected_class = (st.selected_class + 1) % len(CFG["CLASSES"])
                st.status = f"GLOBAL CLASS {st.current_class_name()}"
                st.refresh_folder_count()

        elif key == CFG["KEY_PREV_CLASS"]:
            with st.lock:
                st.selected_class = (st.selected_class - 1) % len(CFG["CLASSES"])
                st.status = f"GLOBAL CLASS {st.current_class_name()}"
                st.refresh_folder_count()

        elif key == CFG["KEY_OVERLAY_MODE"]:
            with st.lock:
                st.overlay_mode = not st.overlay_mode

                if st.overlay_mode:
                    st.status = "OVERLAY MODE ON"
                    sync_overlay_boxes(st)
                else:
                    st.status = "OVERLAY MODE OFF"

                    # ถ้าอยู่ใน freeze mode แล้วปิด overlay ให้กลับไปกรอบปกติ
                    if st.freeze_mode and st.freeze_clean_frame is not None and st.freeze_base_boxes:
                        h, w = st.freeze_clean_frame.shape[:2]
                        current_class_ids = [b["class_id"] for b in st.freeze_boxes] if st.freeze_boxes else [b["class_id"] for b in st.freeze_base_boxes]
                        st.freeze_boxes = rebuild_overlay_boxes(
                            st.freeze_base_boxes,
                            0, 0, 0, 0,
                            w, h,
                            class_ids=current_class_ids
                        )
                    elif not st.freeze_mode:
                        st.overlay_boxes = []

                st.refresh_folder_count()

        # Overlay fine controls (ใช้ได้ทั้ง live และ freeze)
        elif key == CFG["KEY_LEFT_EXPAND"]:
            with st.lock:
                adjust_overlay(st, "left", -CFG["OVERLAY_STEP"])

        elif key == CFG["KEY_LEFT_SHRINK"]:
            with st.lock:
                adjust_overlay(st, "left", +CFG["OVERLAY_STEP"])

        elif key == CFG["KEY_RIGHT_EXPAND"]:
            with st.lock:
                adjust_overlay(st, "right", +CFG["OVERLAY_STEP"])

        elif key == CFG["KEY_RIGHT_SHRINK"]:
            with st.lock:
                adjust_overlay(st, "right", -CFG["OVERLAY_STEP"])

        elif key == CFG["KEY_TOP_EXPAND"]:
            with st.lock:
                adjust_overlay(st, "top", -CFG["OVERLAY_STEP"])

        elif key == CFG["KEY_TOP_SHRINK"]:
            with st.lock:
                adjust_overlay(st, "top", +CFG["OVERLAY_STEP"])

        elif key == CFG["KEY_BOTTOM_EXPAND"]:
            with st.lock:
                adjust_overlay(st, "bottom", +CFG["OVERLAY_STEP"])

        elif key == CFG["KEY_BOTTOM_SHRINK"]:
            with st.lock:
                adjust_overlay(st, "bottom", -CFG["OVERLAY_STEP"])

        elif ord('0') <= key <= ord('0') + len(CFG["CLASSES"]) - 1:
            with st.lock:
                st.selected_class = key - ord('0')
                st.status = f"GLOBAL CLASS {st.current_class_name()}"
                st.refresh_folder_count()

    cv2.destroyAllWindows()

# =========================================================
# CALLBACK
# =========================================================

def callback(element, buffer, st: State):
    if buffer is None:
        return

    fmt, w, h = get_caps_from_pad(element.get_static_pad("src"))
    frame_rgb = get_numpy_from_buffer(buffer, fmt, w, h)
    frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    clean = frame.copy()

    with st.lock:
        st.latest_frame = frame.copy()
        st.latest_clean_frame = clean.copy()

        if not st.freeze_mode:
            new_boxes = []
            detections = hailo.get_roi_from_buffer(buffer).get_objects_typed(hailo.HAILO_DETECTION)

            for d in detections:
                conf = d.get_confidence()
                if conf < CFG["CONF_THRES"]:
                    continue

                b = d.get_bbox()
                x1 = int(b.xmin() * w)
                y1 = int(b.ymin() * h)
                x2 = int(b.xmax() * w)
                y2 = int(b.ymax() * h)

                x1 = clamp(x1, 0, w - 1)
                y1 = clamp(y1, 0, h - 1)
                x2 = clamp(x2, 0, w - 1)
                y2 = clamp(y2, 0, h - 1)

                if x2 <= x1 or y2 <= y1:
                    continue

                new_boxes.append({
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "class_id": st.selected_class,
                })

            st.live_boxes = new_boxes

            if st.overlay_mode:
                st.overlay_boxes = rebuild_overlay_boxes(
                    st.live_boxes,
                    st.overlay_pad_left,
                    st.overlay_pad_right,
                    st.overlay_pad_top,
                    st.overlay_pad_bottom,
                    w, h
                )

        else:
            if st.overlay_mode:
                sync_overlay_boxes(st)

    try:
        if st.freeze_mode and st.freeze_clean_frame is not None:
            st.q.put_nowait(st.freeze_clean_frame.copy())
        else:
            st.q.put_nowait(clean.copy())
    except queue.Full:
        pass

# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    Gst.init(None)
    st = State()

    threading.Thread(target=display_thread, args=(st,), daemon=True).start()

    app = GStreamerDetectionApp(callback, st)
    app.run()
