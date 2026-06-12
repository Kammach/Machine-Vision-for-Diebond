import os, cv2, gi, time, queue, threading, numpy as np, hailo
from datetime import datetime
from collections import defaultdict

os.environ["GST_PLUGIN_FEATURE_RANK"] = "vaapidecodebin:NONE"

gi.require_version("Gst", "1.0")
from gi.repository import Gst

from hailo_apps.python.pipeline_apps.detection.detection_pipeline import GStreamerDetectionApp
from hailo_apps.python.core.common.buffer_utils import get_caps_from_pad, get_numpy_from_buffer
from hailo_apps.python.core.gstreamer.gstreamer_app import app_callback_class


# ====================== CONFIG ======================
CFG = {
    "WIN": "Chip Inspection",
    "NG_SEC": 1.5,
    "MAX_H": 30,
    "SB_W": 500,
    "SHOW": 4,
    "FONT": cv2.FONT_HERSHEY_COMPLEX,
    "C": {
        "OK": (0, 185, 0),
        "NG": (0, 0, 200),
        "BG": (255, 255, 255),
        "HUD": (15, 17, 20),
        "HEAD": (0, 180, 255),
        "CARD": (0, 0, 200),
    }
}


# ====================== STATE ======================
class State(app_callback_class):
    def __init__(self):
        super().__init__()
        self.q = queue.Queue(maxsize=3)
        self.history = []
        self.seen = defaultdict(set)
        self.alerted = {"OK": set(), "NG": set()}
        self.first = {}
        self.ok = 0
        self.ng = 0
        self.offset = 0
        self.lock = threading.Lock()

    def count(self, label, tid):
        if tid < 0 or tid in self.seen[label]:
            return
        self.seen[label].add(tid)
        if label == "OK":
            self.ok += 1
        else:
            self.ng += 1


# ====================== UTILS ======================
C = CFG["C"]
F = CFG["FONT"]


def fit_crop(img, target_h=135, target_w=230):
    if img is None or img.size == 0:
        return np.zeros((target_h, target_w, 3), np.uint8)

    h, w = img.shape[:2]
    r = min(target_w / w, target_h / h)
    rs = cv2.resize(img, (max(1, int(w * r)), max(1, int(h * r))))

    out = np.zeros((target_h, target_w, 3), np.uint8)
    y = (target_h - rs.shape[0]) // 2
    x = (target_w - rs.shape[1]) // 2
    out[y:y + rs.shape[0], x:x + rs.shape[1]] = rs
    return out


def txt(img, text, pos, color=(255, 255, 255), scale=0.9):
    cv2.putText(img, text, pos, F, scale, color, 2, cv2.LINE_8)


# ====================== DRAW ======================
def draw_box(frame, box, label, conf, tid):
    x1, y1, x2, y2 = map(int, box)
    col = C["OK"] if label == "OK" else C["NG"]

    cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
    t = f"{label} {conf:.0%} ID:{tid}"
    y = y1 - 12 if y1 > 40 else y2 + 28

    (tw, _), _ = cv2.getTextSize(t, F, 0.65, 2)
    cv2.rectangle(frame, (x1, y - 28), (x1 + tw + 12, y + 8), col, -1)
    txt(frame, t, (x1 + 5, y), (0, 0, 0), 0.65)


def draw_hud(frame, fps, st):
    h, w = frame.shape[:2]
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (w, 62), C["HUD"], -1)
    cv2.addWeighted(ov, 0.8, frame, 0.2, 0, frame)
    txt(frame, "CHIP INSPECTION", (20, 42), (0, 255, 255), 0.95)
    txt(frame, f"FPS:{fps:.1f}", (w - 190, 42), (220, 220, 220), 0.75)
    txt(frame, f"OK:{st.ok}", (430, 42), C["OK"], 0.85)
    txt(frame, f"NG:{st.ng}", (620, 42), C["NG"], 0.85)


def make_sidebar(st, frame_h, frame, fps):
    w = CFG["SB_W"]
    img = np.full((frame_h, w, 3), C["BG"], np.uint8)

    header_h = 58
    cv2.rectangle(img, (0, 0), (w, header_h), C["HEAD"], -1)
    txt(img, "Results", (18, 40), scale=0.85)
    txt(img, f"FPS:{fps:.1f}", (385, 40), (10, 10, 10), 0.75)
    txt(img, f"OK:{st.ok}", (170, 40), C["OK"], 0.85)
    txt(img, f"NG:{st.ng}", (270, 40), C["NG"], 0.85)

    with st.lock:
        hist = st.history[st.offset: st.offset + CFG["SHOW"]]

    if not hist:
        return img

    num_cards = min(len(hist), CFG["SHOW"])
    available_h = frame_h - header_h - 10
    card_h = max(90, available_h // num_cards - 12)

    for i, e in enumerate(hist):
        y = header_h + 12 + i * (card_h + 12)
        if y + card_h > frame_h - 10:
            break

        card_color = C["OK"] if e["label"] == "OK" else C["NG"]
        cv2.rectangle(img, (12, y), (w - 12, y + card_h), card_color, -1)

        crop_h = int(card_h * 0.88)
        crop = fit_crop(e.get("crop"), target_h=crop_h, target_w=230)

        crop_y = y + (card_h - crop_h) // 2 + 2
        crop_x = w - crop.shape[1] - 18
        img[crop_y:crop_y + crop.shape[0], crop_x:crop_x + crop.shape[1]] = crop

        x_text = 25
        txt(img, f"ID: {e['tid']}", (x_text + 50, y + 32), (255, 255, 100), 0.72)
        txt(img, f"{e['label']}", (x_text, y + 32), (0, 0, 0), 0.69)
        txt(img, f"Conf: {e['conf']:.1%}", (x_text, y + 62), (255, 255, 255), 0.67)
        txt(
            img,
            f"Pos: ({(e['box'][0] + e['box'][2]) // 2}, {(e['box'][1] + e['box'][3]) // 2})",
            (x_text, y + 113),
            (255, 255, 255),
            0.63
        )
        txt(img, e["time"].strftime("%H:%M:%S"), (x_text, y + 87), (220, 220, 220), 0.63)

    return img


# ====================== DISPLAY ======================
def display(st):
    fps = 0.0
    cnt = 0
    t0 = time.time()

    while True:
        cnt += 1
        if time.time() - t0 >= 1.0:
            fps = cnt / (time.time() - t0)
            cnt = 0
            t0 = time.time()

        try:
            frame = st.q.get(timeout=0.04)
        except queue.Empty:
            continue

        sidebar = make_sidebar(st, frame.shape[0], frame, fps)
        ui = np.hstack([frame, sidebar])

        cv2.imshow(CFG["WIN"], ui)
        k = cv2.waitKey(1) & 0xFF

        if k == ord('q') or k == 3:
            break
        elif k in (82, ord('k')):
            st.offset = max(0, st.offset - 1)
        elif k in (84, ord('j')):
            st.offset += 1
        elif k in (ord('c'), ord('C')):
            with st.lock:
                st.history.clear()
                st.alerted["OK"].clear()
                st.alerted["NG"].clear()
                st.first.clear()
                st.seen.clear()
                st.ok = 0
                st.ng = 0
                st.offset = 0

    cv2.destroyAllWindows()


# ====================== CALLBACK ======================
def callback(element, buffer, st):
    if buffer is None:
        return

    fmt, w, h = get_caps_from_pad(element.get_static_pad("src"))

    raw_frame = get_numpy_from_buffer(buffer, fmt, w, h)

    if fmt == "RGB":
        frame = cv2.cvtColor(raw_frame, cv2.COLOR_RGB2BGR)
        orig = frame.copy()
    else:
        frame = raw_frame.copy()
        orig = frame.copy()

    now = time.time()

    for d in hailo.get_roi_from_buffer(buffer).get_objects_typed(hailo.HAILO_DETECTION):
        label = d.get_label()
        if label not in ("OK", "NG"):
            continue

        conf = d.get_confidence()
        b = d.get_bbox()
        box = (int(b.xmin() * w), int(b.ymin() * h), int(b.xmax() * w), int(b.ymax() * h))

        tid_obj = d.get_objects_typed(hailo.HAILO_UNIQUE_ID)
        tid = tid_obj[0].get_id() if tid_obj else -1

        st.count(label, tid)

        if tid > 0:
            if (label, tid) not in st.first:
                st.first[(label, tid)] = now

            if (now - st.first[(label, tid)] >= CFG["NG_SEC"] and tid not in st.alerted[label]):
                st.alerted[label].add(tid)
                x1, y1, x2, y2 = box

                # กัน crop หลุดขอบภาพ
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)

                crop = orig[y1:y2, x1:x2].copy()

                with st.lock:
                    st.history.insert(0, {
                        "time": datetime.now(),
                        "tid": tid,
                        "conf": conf,
                        "box": box,
                        "crop": crop,
                        "label": label
                    })
                    if len(st.history) > CFG["MAX_H"]:
                        st.history.pop()

        draw_box(frame, box, label, conf, tid)

    # เก็บเฟรมล่าสุดไว้เสมอ
    try:
        if st.q.full():
            try:
                st.q.get_nowait()
            except queue.Empty:
                pass
        st.q.put_nowait(frame.copy())
    except queue.Full:
        pass


# ====================== MAIN ======================
if __name__ == "__main__":
    st = State()

    # สร้าง window จาก main thread ก่อน
    cv2.namedWindow(CFG["WIN"], cv2.WINDOW_NORMAL)
    cv2.resizeWindow(CFG["WIN"], 1520, 780)
    cv2.startWindowThread()

    # display อยู่ thread ลูก
    threading.Thread(target=display, args=(st,), daemon=True).start()

    # Hailo/GStreamer ต้องอยู่ main thread เพราะมี signal.signal()
    GStreamerDetectionApp(callback, st).run()
