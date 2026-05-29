import os
import cv2
import gi
import time
import queue
import threading
import numpy as np
import hailo
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
    "NG_SEC": 0.25,
    "MAX_H": 30,
    "SB_W": 500,
    "SHOW": 4,
    "FONT": cv2.FONT_HERSHEY_TRIPLEX,
    "C": {
        "OK": (0, 185, 0),
        "NG": (0, 0, 200),
        "BG": (255, 255, 255),
        "HUD": (15, 17, 20),
        "HEAD": (0, 180, 255),
        "CARD": (0, 0, 200),
        "ROI": (255, 255, 0),
    },
    "ROI_START": 0.4,
    "ROI_END": 0.6,
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


def draw_roi_zone(frame):
    h, w = frame.shape[:2]
    x1 = int(w * CFG["ROI_START"])
    x2 = int(w * CFG["ROI_END"])

    cv2.line(frame, (x1, 0), (x1, h), C["ROI"], 1, cv2.LINE_AA)
    cv2.line(frame, (x2, 0), (x2, h), C["ROI"], 1, cv2.LINE_AA)


def txt(img, text, pos, color=(255, 255, 255), scale=0.9):
    cv2.putText(img, text, pos, F, scale, color, 1, cv2.LINE_AA)
    #cv2.putText(img, text, pos, F, scale, color, thick, cv2.LINE_AA)


def draw_box(frame, box, label, conf, tid, thickness=2):
    x1, y1, x2, y2 = map(int, box)

    if label == "OK":
        col = C["OK"]
    elif label == "NG":
        col = C["NG"]
    else:
        col = (255, 255, 255)

    cv2.rectangle(frame, (x1, y1), (x2, y2), col, thickness, cv2.LINE_8)
    t = f"{label} {conf:.0%} ID:{tid}"
    y = y1 - 12 if y1 > 40 else y2 + 28
    cv2.putText(frame, t, (x1 + 5, y), cv2.FONT_HERSHEY_COMPLEX, 0.65, col, 2, cv2.LINE_8)


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
            0.63,
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

    draw_roi_zone(frame)
    now = time.time()

    roi_x1 = int(w * CFG["ROI_START"])
    roi_x2 = int(w * CFG["ROI_END"])

    for d in hailo.get_roi_from_buffer(buffer).get_objects_typed(hailo.HAILO_DETECTION):
        label = d.get_label()
        if label not in ("OK", "NG"):
            continue

        conf = d.get_confidence()
        b = d.get_bbox()
        box = (int(b.xmin() * w), int(b.ymin() * h), int(b.xmax() * w), int(b.ymax() * h))

        cx = (box[0] + box[2]) // 2
        if not (roi_x1 <= cx <= roi_x2):
            continue

        tid_obj = d.get_objects_typed(hailo.HAILO_UNIQUE_ID)
        tid = tid_obj[0].get_id() if tid_obj else -1

        st.count(label, tid)

        if tid > 0:
            if (label, tid) not in st.first:
                st.first[(label, tid)] = now

            if (now - st.first[(label, tid)] >= CFG["NG_SEC"] and tid not in st.alerted[label]):
                st.alerted[label].add(tid)
                x1, y1, x2, y2 = box
                crop = orig[max(0, y1):max(0, y2), max(0, x1):max(0, x2)].copy()

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

    # GUI ต้องสร้างจาก main thread
    cv2.namedWindow(CFG["WIN"], cv2.WINDOW_NORMAL)
    cv2.resizeWindow(CFG["WIN"], 1520, 780)
    cv2.startWindowThread()

    # display thread
    display_thread = threading.Thread(
        target=display,
        args=(st,),
        daemon=True
    )
    display_thread.start()

    # Hailo app main thread
    app = GStreamerDetectionApp(callback, st)
    app.run()


"""
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
    "NG_SEC": 0.25,            # ไม่ใช้ delay ใน alert ใหม่ ยังคงเผื่อไว้
    "MAX_H": 30,
    "SB_W": 500,
    "SHOW": 4,
    "FONT": cv2.FONT_HERSHEY_COMPLEX,
    "C": {
        "OK": (0, 185 , 0),         # สีเขียวอ่อนสำหรับ OK card
        "NG": (0, 0, 200),             # สีแดงเข้มสำหรับ NG card
        "BG": (255, 255, 255),
        "HUD": (15, 17, 20),
        "HEAD": (0, 180, 255),
        "CARD": (0, 0, 200),
        "ROI": (255, 255, 0), # สีฟ้า/น้ำเงินอ่อน สำหรับเส้นคอลัมน์ ROI
    },

        # === [เพิ่มตรงนี้] กำหนดพื้นที่คอลัมน์แนวตั้ง (ใช้เป็น % ของความกว้างภาพ 0.0 - 1.0) ===
    "ROI_START": 0.4,  # เริ่มต้นที่ 40% ของความกว้างภาพ
    "ROI_END": 0.6,    # สิ้นสุดที่ 60% ของความกว้างภาพ
}

# ====================== STATE ======================
class State(app_callback_class):
    def __init__(self):
        super().__init__()
        self.q = queue.Queue(3)
        self.history = []
        self.seen = defaultdict(set)
        self.alerted = {"OK": set(), "NG": set()}   # แยกเก็บ alert ตาม label
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
    rs = cv2.resize(img, (int(w * r), int(h * r)))

    out = np.zeros((target_h, target_w, 3), np.uint8)
    y = (target_h - rs.shape[0]) // 2
    x = (target_w - rs.shape[1]) // 2
    out[y:y + rs.shape[0], x:x + rs.shape[1]] = rs
    return out   

def draw_roi_zone(frame):
    h, w = frame.shape[:2]
    x1 = int(w * CFG["ROI_START"])
    x2 = int(w * CFG["ROI_END"])
    
    # วาดเส้นแนวตั้งซ้าย-ขวา เพื่อแสดงกรอบคอลัมน์
    cv2.line(frame, (x1, 0), (x1, h), C["ROI"], 1, cv2.LINE_AA)
    cv2.line(frame, (x2, 0), (x2, h), C["ROI"], 1, cv2.LINE_AA)
    
    '''# ทำไฮไลต์โปร่งแสงจางๆ ในโซนตรวจจับ
    sub_img = frame[:, x1:x2]
    overlay = sub_img.copy()
    cv2.rectangle(overlay, (0, 0), (sub_img.shape[1], sub_img.shape[0]), C["ROI"], -1)
    frame[:, x1:x2] = cv2.addWeighted(overlay, 0.1, sub_img, 0.9, 0)
    '''
    # ข้อความกำกับโซน
    #txt(frame, "DETECTION ZONE", (x1 + 10, h - 20), C["ROI"], 0.5)


def txt(img, text, pos, color=(255, 255, 255), scale=0.9):
    cv2.putText(img, text, pos, F, scale, color, 2, cv2.LINE_8)


def draw_box(frame, box, label, conf, tid, thickness=2):
    x1, y1, x2, y2 = map(int, box)
    if label == "OK":
        col = C["OK"]
    elif label == "NG":
        col = C["NG"]
    else:
        col = (255, 255, 255)
    cv2.rectangle(frame, (x1, y1), (x2, y2), col, thickness, cv2.LINE_8)
    t = f"{label} {conf:.0%} ID:{tid}"
    y = y1 - 12 if y1 > 40 else y2 + 28
    #(tw, _), _ = cv2.getTextSize(t, F, 0.65, 2)
    #cv2.rectangle(frame, (x1, y-28), (x1 + tw + 12, y + 8), col, -1)
    #txt(frame, t, (x1 + 5, y), (0, 0, 0), 0.65)
    cv2.putText(frame, t, (x1 + 5, y),cv2.FONT_HERSHEY_COMPLEX , 0.65, col, 2, cv2.LINE_8)    

def draw_hud(frame, fps, st):
    h, w = frame.shape[:2]
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (w, 62), C["HUD"], -1)
    cv2.addWeighted(ov, 0.8, frame, 0.2, 0, frame)
    txt(frame, "CHIP INSPECTION", (20, 42), (0, 255, 255), 0.95)
    txt(frame, f"FPS:{fps:.1f}", (w-190, 42), (220, 220, 220), 0.75)
    txt(frame, f"OK:{st.ok}", (430, 42), C["OK"], 0.85)
    txt(frame, f"NG:{st.ng}", (620, 42), C["NG"], 0.85)


def make_sidebar(st, frame_h, frame, fps):
    w = CFG["SB_W"]
    img = np.full((frame_h, w, 3), C["BG"], np.uint8)

    # ==================== Header ====================
    header_h = 58
    cv2.rectangle(img, (0, 0), (w, header_h), C["HEAD"], -1)
    txt(img, "Results ", (18, 40), scale=0.85)
    txt(img, f"FPS:{fps:.1f}", (385, 40), (10, 10, 10), 0.75)
    txt(img, f"OK:{st.ok}", (170, 40), C["OK"], 0.85)
    txt(img, f"NG:{st.ng}", (270, 40), C["NG"], 0.85)

    # ==================== History ====================
    with st.lock:
        hist = st.history[st.offset : st.offset + CFG["SHOW"]]

    if not hist:
        return img

    num_cards = min(len(hist), CFG["SHOW"])   # สูงสุด 4

    # คำนวณความสูงการ์ดให้เหมาะสมกับ 4 การ์ด
    available_h = frame_h - header_h - 10                    # เว้นขอบบน-ล่าง
    card_h = max(90, available_h // num_cards - 12)         # อย่างน้อย 90 px

    for i, e in enumerate(hist):
        y = header_h + 12 + i * (card_h + 12)

        if y + card_h > frame_h - 10:   # ป้องกันล้น
            break

        # การ์ด
        card_color = C["OK"] if e["label"] == "OK" else C["NG"]
        cv2.rectangle(img, (12, y), (w - 12, y + card_h), card_color, -1)

        # ==================== Crop (Dynamic) ====================
        crop_h = int(card_h * 0.88)           # ใช้ประมาณ 68% ของความสูงการ์ด (ใกล้เคียงเดิม)
        crop = fit_crop(e.get("crop"), target_h=crop_h, target_w=230)

        crop_y = y + (card_h - crop_h) // 2 + 2
        crop_x = w - crop.shape[1] - 18
        img[crop_y:crop_y + crop.shape[0], crop_x:crop_x + crop.shape[1]] = crop

        # ==================== Text ====================
        x_text = 25
        txt(img, f"ID: {e['tid']}",        (x_text + 50, y + 32), (255, 255, 100), 0.72)
        txt(img, f"{e['label']}",          (x_text, y + 32), (0, 0, 0),       0.69)
        txt(img, f"Conf: {e['conf']:.1%}", (x_text, y + 62), (255, 255, 255), 0.67)
        txt(img, f"Pos: ({(e['box'][0]+e['box'][2])//2}, {(e['box'][1]+e['box'][3])//2})", 
                                           (x_text, y + 113), (255, 255, 255), 0.63)
        txt(img, e["time"].strftime("%H:%M:%S"), 
                                           (x_text, y + 87), (220, 220, 220), 0.63)

    return img

 
    


# ====================== DISPLAY ======================
def display(st):
    cv2.namedWindow(CFG["WIN"], cv2.WINDOW_NORMAL)
    cv2.resizeWindow(CFG["WIN"], 1520, 780)
    fps = cnt = 0
    t0 = time.time()

    while True:
        cnt += 1
        if time.time() - t0 >= 1:
            fps = cnt / (time.time() - t0)
            cnt, t0 = 0, time.time()

        try: frame = st.q.get(timeout=0.04)
        except queue.Empty: continue

        ui = np.hstack([frame, make_sidebar(st, frame.shape[0], frame, fps)])
        cv2.imshow(CFG["WIN"], ui)
        k = cv2.waitKey(1) & 0xFF

        if k == ord('q') or k == 3: break
        elif k in (82, ord('k')): st.offset = max(0, st.offset - 1)
        elif k in (84, ord('j')): st.offset += 1
        elif k in (ord('c'), ord('C')):
            with st.lock:
                st.history.clear()
                st.alerted["OK"].clear()
                st.alerted["NG"].clear()
                st.first.clear()
                st.seen.clear()
                st.ok = st.ng = st.offset = 0

    cv2.destroyAllWindows()

'''
def display(st): #----------------------------------free size
    win = CFG["WIN"]
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_FREERATIO)
    cv2.resizeWindow(win, 800, 600)
    fps = cnt = 0
    t0 = time.time()
    
    while True:
        cnt += 1
        if time.time() - t0 >= 1:
            fps = cnt / (time.time() - t0)
            cnt = 0
            t0 = time.time()
            
        try:
            frame = st.q.get(timeout=0.04)
        except queue.Empty:
            continue
            
        target_h = 900
        target_w = int(target_h * 4 / 3)
        
        video = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        sidebar = make_sidebar(st, target_h, frame, fps)
        
        ui = np.hstack([video, sidebar])
'''        

# ====================== CALLBACK ======================
def callback(element, buffer, st):
    if buffer is None: 
        return

    fmt, w, h = get_caps_from_pad(element.get_static_pad("src"))
    
    # ดึง frame ดิบจาก GStreamer
    raw_frame = get_numpy_from_buffer(buffer, fmt, w, h)
    frame = raw_frame.copy()

    if fmt == "RGB":
        frame = cv2.cvtColor(raw_frame, cv2.COLOR_RGB2BGR)
        orig = frame.copy()
    else:
        frame = raw_frame.copy()  
        orig = frame.copy()  

    # === [เพิ่มตรงนี้] วาดกรอบคอลัมน์ให้เห็นบนจอภาพ ===
    draw_roi_zone(frame)
    
    now = time.time()

    for d in hailo.get_roi_from_buffer(buffer).get_objects_typed(hailo.HAILO_DETECTION):
        label = d.get_label()
        if label not in ("OK", "NG"): 
            continue

        conf = d.get_confidence()
        b = d.get_bbox()
        box = (int(b.xmin()*w), int(b.ymin()*h), int(b.xmax()*w), int(b.ymax()*h))

        # ถ้าจุดกึ่งกลางของชิป "ไม่อยู่" ในช่วงคอลัมน์ที่กำหนด -> ให้ข้ามไปเลย ไม่นับ ไม่วาดกรอบ
                # ROI CHECK
        cx = (box[0] + box[2]) // 2
        roi_x1 = int(w * CFG["ROI_START"])
        roi_x2 = int(w * CFG["ROI_END"])

        if not (roi_x1 <= cx <= roi_x2):
            continue

        tid_obj = d.get_objects_typed(hailo.HAILO_UNIQUE_ID)
        tid = tid_obj[0].get_id() if tid_obj else -1

        st.count(label, tid)

        if tid > 0:
            if (label, tid) not in st.first:
                st.first[(label, tid)] = now

            if (now - st.first[(label, tid)] >= CFG["NG_SEC"] and
                    tid not in st.alerted[label]):
                st.alerted[label].add(tid)
                x1, y1, x2, y2 = box
                crop = orig[y1:y2, x1:x2].copy()   # crop จาก RGB

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

        draw_box(frame, box, label, conf, tid)   # วาดบน RGB

    # ส่ง frame ที่เป็น RGB ไปให้ display
    try:
        st.q.put_nowait(frame.copy())
    except queue.Full:
        pass
        

# ====================== MAIN ======================
if __name__ == "__main__":
    st = State()
    threading.Thread(target=display, args=(st,), daemon=True).start()
    GStreamerDetectionApp(callback, st).run() 


"""






'''
#GEM

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
    "NG_SEC": 0.25,            
    "MAX_H": 30,
    "SB_W": 300,               # [แก้] ลดความกว้าง Sidebar ลงเพื่อให้ประมวลผลไวขึ้น
    "SHOW": 4,
    "FONT": cv2.FONT_HERSHEY_COMPLEX,
    "C": {
        "OK": (0, 185 , 0),         
        "NG": (0, 0, 200),              
        "BG": (255, 255, 255),
        "HUD": (15, 17, 20),
        "HEAD": (0, 180, 255),
        "CARD": (0, 0, 200),
        "ROI": (255, 255, 0), 
    },
    "ROI_START": 0.4,  
    "ROI_END": 0.6,    
}

# ====================== STATE ======================
class State(app_callback_class):
    def __init__(self):
        super().__init__()
        self.q = queue.Queue(3)
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
    rs = cv2.resize(img, (int(w * r), int(h * r)))

    out = np.zeros((target_h, target_w, 3), np.uint8)
    y = (target_h - rs.shape[0]) // 2
    x = (target_w - rs.shape[1]) // 2
    out[y:y + rs.shape[0], x:x + rs.shape[1]] = rs
    return out   

def draw_roi_zone(frame):
    h, w = frame.shape[:2]
    x1 = int(w * CFG["ROI_START"])
    x2 = int(w * CFG["ROI_END"])
    
    # [แก้] วาดแค่เส้นขอบ ปิดระบบ Alpha Blending (โปร่งแสง) ไปเลยเพื่อรีด FPS
    cv2.line(frame, (x1, 0), (x1, h), C["ROI"], 2, cv2.LINE_AA)
    cv2.line(frame, (x2, 0), (x2, h), C["ROI"], 2, cv2.LINE_AA)
    
    # ข้อความกำกับโซน
    txt(frame, "DETECTION ZONE", (x1 + 10, h - 20), C["ROI"], 0.5)

def txt(img, text, pos, color=(255, 255, 255), scale=0.9):
    cv2.putText(img, text, pos, F, scale, color, 2, cv2.LINE_8)

def draw_box(frame, box, label, conf, tid, thickness=2):
    x1, y1, x2, y2 = map(int, box)
    if label == "OK":
        col = C["OK"]
    elif label == "NG":
        col = C["NG"]
    else:
        col = (255, 255, 255)
    cv2.rectangle(frame, (x1, y1), (x2, y2), col, thickness, cv2.LINE_8)
    t = f"{label} {conf:.0%} ID:{tid}"
    y = y1 - 12 if y1 > 40 else y2 + 28
    cv2.putText(frame, t, (x1 + 5, y), cv2.FONT_HERSHEY_COMPLEX , 0.65, col, 2, cv2.LINE_8)    

def make_sidebar(st, frame_h, frame, fps):
    w = CFG["SB_W"]
    img = np.full((frame_h, w, 3), C["BG"], np.uint8)

    # ==================== Header ====================
    header_h = 58
    cv2.rectangle(img, (0, 0), (w, header_h), C["HEAD"], -1)
    # [แก้] ปรับตำแหน่งแกน X ของตัวหนังสือให้พอดีกับความกว้าง 300px
    txt(img, "Result", (10, 40), scale=0.75)
    txt(img, f"OK:{st.ok}", (110, 40), C["OK"], 0.75)
    txt(img, f"NG:{st.ng}", (190, 40), C["NG"], 0.75)
    txt(img, f"{fps:.0f} FPS", (245, 18), (10, 10, 10), 0.45)

    # ==================== History ====================
    with st.lock:
        hist = st.history[st.offset : st.offset + CFG["SHOW"]]

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

        # ==================== Crop ====================
        crop_h = int(card_h * 0.88)
        # [แก้] ย่อขนาดเป้าหมายการคลอปลงให้พอดีกับ Sidebar เล็ก
        crop = fit_crop(e.get("crop"), target_h=crop_h, target_w=140)

        crop_y = y + (card_h - crop_h) // 2 + 2
        crop_x = w - crop.shape[1] - 18
        img[crop_y:crop_y + crop.shape[0], crop_x:crop_x + crop.shape[1]] = crop

        # ==================== Text ====================
        x_text = 18
        txt(img, f"ID: {e['tid']}",        (x_text + 45, y + 25), (255, 255, 100), 0.6)
        txt(img, f"{e['label']}",          (x_text, y + 25), (0, 0, 0),       0.6)
        txt(img, f"Conf: {e['conf']:.1%}", (x_text, y + 50), (255, 255, 255), 0.55)
        txt(img, f"Pos: ({(e['box'][0]+e['box'][2])//2}, {(e['box'][1]+e['box'][3])//2})", 
                                           (x_text, y + 95), (255, 255, 255), 0.5)
        txt(img, e["time"].strftime("%H:%M:%S"), 
                                           (x_text, y + 72), (220, 220, 220), 0.5)

    return img


# ====================== DISPLAY ======================
def display(st):
    # [แก้] เปิดใช้งาน OpenGL เพื่อใช้การ์ดจอช่วยวาดหน้าต่าง (ถ้ามีปัญหาจอดำให้ลบ | cv2.WINDOW_OPENGL ออก)
    cv2.namedWindow(CFG["WIN"], cv2.WINDOW_NORMAL | cv2.WINDOW_OPENGL)
    cv2.resizeWindow(CFG["WIN"], 1280, 720)
    fps = cnt = 0
    t0 = time.time()

    while True:
        cnt += 1
        if time.time() - t0 >= 1:
            fps = cnt / (time.time() - t0)
            cnt, t0 = 0, time.time()

        try:
            frame = st.q.get(timeout=0.04)
        except queue.Empty:
            continue

        ui = np.hstack([frame, make_sidebar(st, frame.shape[0], frame, fps)])
        cv2.imshow(CFG["WIN"], ui)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q') or k == 3:
            break
        elif k in (82, ord('k')):      # Up
            st.offset = max(0, st.offset - 1)
        elif k in (84, ord('j')):      # Down
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

    fmt, orig_w, orig_h = get_caps_from_pad(element.get_static_pad("src"))
    
    # ดึง frame ดิบจาก GStreamer
    raw_frame = get_numpy_from_buffer(buffer, fmt, orig_w, orig_h)
    
    # === [แก้] ย่อภาพทันที เอาให้เล็กและลื่น ===
    WORK_W = 800  # ถ้ายังช้าไป ลดเหลือ 640 ได้เลย ภาพแตกหน่อยแต่ไวแน่นอน
    scale = WORK_W / orig_w
    WORK_H = int(orig_h * scale)
    
    # รีไซซ์ด้วย INTER_NEAREST (เร็วที่สุด)
    frame = cv2.resize(raw_frame, (WORK_W, WORK_H), interpolation=cv2.INTER_NEAREST)

    if fmt == "RGB":
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        orig = frame.copy()
    else:
        orig = frame.copy()  

    w, h = WORK_W, WORK_H

    # วาดกรอบคอลัมน์ให้เห็นบนจอภาพ
    draw_roi_zone(frame)
    
    now = time.time()

    for d in hailo.get_roi_from_buffer(buffer).get_objects_typed(hailo.HAILO_DETECTION):
        label = d.get_label()
        if label not in ("OK", "NG"): 
            continue

        conf = d.get_confidence()
        b = d.get_bbox()
        
        # จุดนี้จะคูณด้วย w, h ที่เป็นขนาดภาพเล็กแล้ว
        box = (int(b.xmin()*w), int(b.ymin()*h), int(b.xmax()*w), int(b.ymax()*h))

        cx = (box[0] + box[2]) // 2
        roi_x1 = int(w * CFG["ROI_START"])
        roi_x2 = int(w * CFG["ROI_END"])

        if not (roi_x1 <= cx <= roi_x2):
            continue

        tid_obj = d.get_objects_typed(hailo.HAILO_UNIQUE_ID)
        tid = tid_obj[0].get_id() if tid_obj else -1

        st.count(label, tid)

        if tid > 0:
            if (label, tid) not in st.first:
                st.first[(label, tid)] = now

            if (now - st.first[(label, tid)] >= CFG["NG_SEC"] and tid not in st.alerted[label]):
                st.alerted[label].add(tid)
                x1, y1, x2, y2 = box
                
                # คลอปจากภาพที่ย่อแล้ว (ระวังเรื่องขอบภาพลบ)
                crop = orig[max(0, y1):y2, max(0, x1):x2].copy()

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

        # วาดกล่อง ลดความหนาเส้นลงเหลือ 1 เพื่อความไว
        draw_box(frame, box, label, conf, tid, thickness=1)   

    try:
        # ส่งภาพไปคิวโดยไม่ต้อง .copy() แล้ว ช่วยประหยัดแรมและ CPU
        st.q.put_nowait(frame)
    except queue.Full:
        pass
        

# ====================== MAIN ======================
if __name__ == "__main__":
    st = State()
    threading.Thread(target=display, args=(st,), daemon=True).start()
    GStreamerDetectionApp(callback, st).run()
'''


#---------------------------------------------------------------------------------------------------------------------------------------------


'''
#CLAUDE
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
    "NG_SEC": 0.25,
    "MAX_H": 30,
    "SB_W": 500,
    "SHOW": 4,
    "FONT": cv2.FONT_HERSHEY_COMPLEX,

    # ✅ [FPS] ลด resolution ก่อน imshow — หน้าต่างขยายเองอัตโนมัติ
    # ลองค่า 0.5 ก่อน, ถ้ายังไม่เร็วพอลองลด 0.35 หรือ 0.25
    "RENDER_SCALE": 0.5,

    "C": {
        "OK":   (0, 185, 0),
        "NG":   (0, 0, 200),
        "BG":   (255, 255, 255),
        "HUD":  (15, 17, 20),
        "HEAD": (0, 180, 255),
        "CARD": (0, 0, 200),
        "ROI":  (255, 255, 0),
    },

    "ROI_START": 0.4,
    "ROI_END":   0.6,
}

# ====================== STATE ======================
class State(app_callback_class):
    def __init__(self):
        super().__init__()
        self.q = queue.Queue(3)
        self.history = []
        self.seen = defaultdict(set)
        self.alerted = {"OK": set(), "NG": set()}
        self.first = {}
        self.ok = 0
        self.ng = 0
        self.offset = 0
        self.lock = threading.Lock()

        # ✅ [FPS] Cache sidebar — rebuild เฉพาะเมื่อ history หรือ offset เปลี่ยน
        self._sb_cache = None
        self._sb_cache_key = None   # (len(history), offset, ok, ng)

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
    # ✅ [FPS] INTER_NEAREST เร็วกว่า default (INTER_LINEAR) มาก
    rs = cv2.resize(img, (int(w * r), int(h * r)), interpolation=cv2.INTER_NEAREST)
    out = np.zeros((target_h, target_w, 3), np.uint8)
    y = (target_h - rs.shape[0]) // 2
    x = (target_w - rs.shape[1]) // 2
    out[y:y + rs.shape[0], x:x + rs.shape[1]] = rs
    return out


def draw_roi_zone(frame):
    h, w = frame.shape[:2]
    x1 = int(w * CFG["ROI_START"])
    x2 = int(w * CFG["ROI_END"])
    cv2.line(frame, (x1, 0), (x1, h), C["ROI"], 1, cv2.LINE_AA)
    cv2.line(frame, (x2, 0), (x2, h), C["ROI"], 1, cv2.LINE_AA)
    sub_img = frame[:, x1:x2]
    overlay = sub_img.copy()
    cv2.rectangle(overlay, (0, 0), (sub_img.shape[1], sub_img.shape[0]), C["ROI"], -1)
    frame[:, x1:x2] = cv2.addWeighted(overlay, 0.1, sub_img, 0.9, 0)
    txt(frame, "DETECTION ZONE", (x1 + 10, h - 20), C["ROI"], 0.5)


def txt(img, text, pos, color=(255, 255, 255), scale=0.9):
    cv2.putText(img, text, pos, F, scale, color, 2, cv2.LINE_8)


def draw_box(frame, box, label, conf, tid, thickness=2):
    x1, y1, x2, y2 = map(int, box)
    col = C.get(label, (255, 255, 255))
    cv2.rectangle(frame, (x1, y1), (x2, y2), col, thickness, cv2.LINE_8)
    t = f"{label} {conf:.0%} ID:{tid}"
    y = y1 - 12 if y1 > 40 else y2 + 28
    cv2.putText(frame, t, (x1 + 5, y), cv2.FONT_HERSHEY_COMPLEX, 0.65, col, 2, cv2.LINE_8)


def make_sidebar(st, frame_h, fps):
    """สร้าง sidebar — เรียกผ่าน get_sidebar_cached เพื่อใช้ cache"""
    w = CFG["SB_W"]
    img = np.full((frame_h, w, 3), C["BG"], np.uint8)

    # Header
    header_h = 58
    cv2.rectangle(img, (0, 0), (w, header_h), C["HEAD"], -1)
    txt(img, "Results",          (18, 40),  scale=0.85)
    txt(img, f"FPS:{fps:.1f}",   (385, 40), (10, 10, 10), 0.75)
    txt(img, f"OK:{st.ok}",      (170, 40), C["OK"],      0.85)
    txt(img, f"NG:{st.ng}",      (270, 40), C["NG"],      0.85)

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
        txt(img, f"ID: {e['tid']}",        (x_text + 50, y + 32), (255, 255, 100), 0.72)
        txt(img, f"{e['label']}",          (x_text,      y + 32), (0, 0, 0),       0.69)
        txt(img, f"Conf: {e['conf']:.1%}", (x_text,      y + 62), (255, 255, 255), 0.67)
        txt(img, f"Pos: ({(e['box'][0]+e['box'][2])//2}, {(e['box'][1]+e['box'][3])//2})",
                                           (x_text,      y + 113),(255, 255, 255), 0.63)
        txt(img, e["time"].strftime("%H:%M:%S"),
                                           (x_text,      y + 87), (220, 220, 220), 0.63)

    return img


def get_sidebar_cached(st, frame_h, fps):
    """
    ✅ [FPS] คืน sidebar จาก cache ถ้าข้อมูลไม่เปลี่ยน
    rebuild เฉพาะเมื่อ history, offset, หรือ counter เปลี่ยน
    """
    with st.lock:
        cache_key = (len(st.history), st.offset, st.ok, st.ng)

    if st._sb_cache is not None and st._sb_cache_key == cache_key:
        return st._sb_cache

    sidebar = make_sidebar(st, frame_h, fps)
    st._sb_cache = sidebar
    st._sb_cache_key = cache_key
    return sidebar


# ====================== DISPLAY ======================
def display(st):
    cv2.namedWindow(CFG["WIN"], cv2.WINDOW_NORMAL)
    cv2.resizeWindow(CFG["WIN"], 1520, 780)

    fps = cnt = 0
    t0 = time.time()
    scale = CFG.get("RENDER_SCALE", 0.5)

    while True:
        cnt += 1
        if time.time() - t0 >= 1:
            fps = cnt / (time.time() - t0)
            cnt, t0 = 0, time.time()

        try:
            frame = st.q.get(timeout=0.04)
        except queue.Empty:
            continue

        # ✅ [FPS] ใช้ sidebar cache — ไม่ rebuild ทุก frame
        sidebar = get_sidebar_cached(st, frame.shape[0], fps)
        ui = np.hstack([frame, sidebar])

        # ✅ [FPS] Resize ให้เล็กก่อน imshow
        #    cv2.WINDOW_NORMAL จะ stretch ให้เต็มหน้าต่างเองโดยอัตโนมัติ
        if scale != 1.0:
            h, w = ui.shape[:2]
            ui_small = cv2.resize(
                ui,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_NEAREST,   # เร็วสุด, ภาพแตกได้ไม่สน
            )
        else:
            ui_small = ui

        cv2.imshow(CFG["WIN"], ui_small)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q') or k == 3:
            break
        elif k in (82, ord('k')):       # Up
            st.offset = max(0, st.offset - 1)
        elif k in (84, ord('j')):       # Down
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
            # ✅ bust cache หลัง clear
            st._sb_cache = None
            st._sb_cache_key = None

    cv2.destroyAllWindows()


# ====================== CALLBACK ======================
def callback(element, buffer, st):
    if buffer is None:
        return

    fmt, w, h = get_caps_from_pad(element.get_static_pad("src"))
    raw_frame = get_numpy_from_buffer(buffer, fmt, w, h)
    frame = raw_frame.copy()

    if fmt == "RGB":
        frame = cv2.cvtColor(raw_frame, cv2.COLOR_RGB2BGR)
        orig = frame.copy()
    else:
        frame = raw_frame.copy()
        orig = frame.copy()

    draw_roi_zone(frame)

    now = time.time()
    roi_x1 = int(w * CFG["ROI_START"])
    roi_x2 = int(w * CFG["ROI_END"])

    for d in hailo.get_roi_from_buffer(buffer).get_objects_typed(hailo.HAILO_DETECTION):
        label = d.get_label()
        if label not in ("OK", "NG"):
            continue

        conf = d.get_confidence()
        b = d.get_bbox()
        box = (int(b.xmin()*w), int(b.ymin()*h), int(b.xmax()*w), int(b.ymax()*h))

        cx = (box[0] + box[2]) // 2
        if not (roi_x1 <= cx <= roi_x2):
            continue

        tid_obj = d.get_objects_typed(hailo.HAILO_UNIQUE_ID)
        tid = tid_obj[0].get_id() if tid_obj else -1

        st.count(label, tid)

        if tid > 0:
            if (label, tid) not in st.first:
                st.first[(label, tid)] = now

            if (now - st.first[(label, tid)] >= CFG["NG_SEC"] and
                    tid not in st.alerted[label]):
                st.alerted[label].add(tid)
                x1, y1, x2, y2 = box
                crop = orig[y1:y2, x1:x2].copy()

                with st.lock:
                    st.history.insert(0, {
                        "time": datetime.now(),
                        "tid": tid,
                        "conf": conf,
                        "box": box,
                        "crop": crop,
                        "label": label,
                    })
                    if len(st.history) > CFG["MAX_H"]:
                        st.history.pop()

        draw_box(frame, box, label, conf, tid)

    try:
        st.q.put_nowait(frame.copy())
    except queue.Full:
        pass


# ====================== MAIN ======================
if __name__ == "__main__":
    st = State()
    threading.Thread(target=display, args=(st,), daemon=True).start()
    GStreamerDetectionApp(callback, st).run()
''' 
