# Dataset Preparing — Machine Vision for Diebond

เครื่องมือเตรียม Dataset สำหรับ YOLO Object Detection ในงาน Machine Vision ตรวจสอบ Diebond  
รองรับการเก็บภาพจากกล้อง (Raspberry Pi / Hailo), การ Label, การจัดการ Dataset และการแบ่ง Train/Val

---

## สารบัญ

- [ภาพรวม](#ภาพรวม)
- [โครงสร้างไฟล์](#โครงสร้างไฟล์)
- [รายละเอียดแต่ละสคริปต์](#รายละเอียดแต่ละสคริปต์)
  - [Capture_Labeling_Tool.py](#1-capture_labeling_toolpy--raspberry-pi)
  - [Image_Labeling_Tool.py](#2-image_labeling_toolpy--labeling-จากภาพที่มีอยู่แล้ว)
  - [easy_to_collect_update.py](#3-easy_to_collect_updatepy--hailo-ai-accelerator)
  - [dataset_tools.py](#4-dataset_toolspy--เครื่องมือจัดการ-dataset)
  - [split_dataset.py](#5-split_datasetpy--แบ่งข้อมูล-trainval)
- [รูปแบบ Dataset (YOLO Format)](#รูปแบบ-dataset-yolo-format)
- [โครงสร้างโฟลเดอร์ Output](#โครงสร้างโฟลเดอร์-output)
- [Dependencies](#dependencies)
- [ข้อควรระวัง](#ข้อควรระวัง)

---

## ภาพรวม

โฟลเดอร์นี้รวมสคริปต์สำหรับ **Pipeline การเตรียม Dataset** ตั้งแต่ต้นจนจบ:

```
เก็บภาพจากกล้อง  →  Label Box  →  จัดการ/Merge Dataset  →  แบ่ง Train/Val
```

Label ทุกไฟล์ใช้รูปแบบ **YOLO Normalized Format** (`.txt`)  
ชื่อไฟล์ภาพและ label ตรงกันทุกคู่: `img_XXXX.jpg` + `img_XXXX.txt`

**Class ที่ใช้ในโปรเจกต์:**

| Class ID | ชื่อ | ความหมาย |
|----------|------|-----------|
| 0 | NG | No Good (ไม่มีชิป) |
| 1 | OK | Good (มีชิป) |

> **หมายเหตุ:** `easy_to_collect_update.py` ใช้ 7 classes (OK, NG, GL, MA, DF, OG, class6) ซึ่งมากกว่าสคริปต์อื่น

---

## โครงสร้างไฟล์

```
Dataset Preparing/
├── Capture_Labeling_Tool.py      # เก็บภาพจาก Raspberry Pi Camera + Label พร้อมกัน
├── Image_Labeling_Tool.py        # Label ภาพที่มีอยู่แล้วใน disk
├── easy_to_collect_update.py     # เก็บภาพ + Label บน Hailo AI Accelerator
├── dataset_tools.py              # เครื่องมือจัดการ Dataset (Merge, Relabel, Validate ฯลฯ)
├── split_dataset.py              # แบ่ง Dataset เป็น Train/Val
└── README.md
```

---

## รายละเอียดแต่ละสคริปต์

### 1. `Capture_Labeling_Tool.py` — Raspberry Pi
<img width="1044" height="894" alt="image" src="https://github.com/user-attachments/assets/72554c83-3017-4fb3-b760-a436900ac18e" />


**วัตถุประสงค์:** เก็บภาพจากกล้อง Raspberry Pi (ผ่าน `picamera2`) พร้อม Label ในขั้นตอนเดียว

**Hardware ที่ต้องการ:** Raspberry Pi + PiCamera

**วิธีใช้:**

```bash
python Capture_Labeling_Tool.py
```

**ฟีเจอร์หลัก:**
- แสดง Live Preview ขนาด 2028×1520 px
- กด `F` เพื่อ Freeze frame แล้วเริ่ม Label
- รองรับ 2 วิธีวาด Bounding Box:
  - **คลิกครั้งเดียว** → สร้าง Box ขนาดที่กำหนดไว้ตรงจุดที่คลิก
  - **Drag-draw mode** (กด `D`) → ลากเมาส์วาด Box อิสระ
- บันทึกเป็น `.jpg` + `.txt` (YOLO format) ที่โฟลเดอร์ `dataset/`

**Keyboard Shortcuts:**

| ปุ่ม | การทำงาน |
|------|----------|
| `F` | Freeze / Unfreeze frame |
| `S` | บันทึก frame + label ปัจจุบัน |
| `M` | สลับ Class (NG ↔ OK) |
| `D` | เปิด/ปิด Drag-draw mode (ต้อง Freeze ก่อน) |
| `Z` | ลบ Box ล่าสุด |
| `C` | ลบ Box ทั้งหมดใน frame นี้ |
| `+` / `=` | ขยาย Box (กว้าง+สูง) |
| `-` | ลด Box (กว้าง+สูง) |
| `[` / `]` | ลด/ขยายเฉพาะความกว้าง |
| `;` / `'` | ขยาย/ลดเฉพาะความสูง |
| `R` | Reset ขนาด Box เป็นค่าเริ่มต้น (300×300) |
| `Q` / `ESC` | ออกจากโปรแกรม |

**ค่าที่แก้ได้ในโค้ด (ส่วน EASY TO EDIT SETTINGS):**

```python
DATASET_ROOT = "dataset"          # โฟลเดอร์บันทึก
INITIAL_BOX_WIDTH = 300           # ขนาด Box เริ่มต้น
INITIAL_BOX_HEIGHT = 300
PREVIEW_WIDTH = 2028              # ความละเอียด Preview
PREVIEW_HEIGHT = 1520
FILE_INDEX_WIDTH = 4              # ความกว้างของเลขไฟล์ (img_0001)
CLASS_NAMES = {0: "NG", 1: "OK"} # ชื่อ Class
```

---

### 2. `Image_Labeling_Tool.py` — Labeling จากภาพที่มีอยู่แล้ว

**วัตถุประสงค์:** Label Bounding Box บนภาพที่เก็บไว้ใน disk แล้ว (ไม่ต้องใช้กล้อง)

**วิธีใช้:**

```bash
python Image_Labeling_Tool.py
```

> ⚠️ **ต้องแก้ Path ในโค้ดก่อนใช้:**
> ```python
> IMG_DIR = r"D:\March\MCphase3\chip\dataset_raw\images"  # แก้ให้ตรงกับ path ของคุณ
> LBL_DIR = r"D:\March\MCphase3\chip\dataset_raw\labels"
> ```
<img width="673" height="50" alt="image" src="https://github.com/user-attachments/assets/dd27b091-a105-4469-b459-e4d353422c7a" />
<img width="1137" height="664" alt="image" src="https://github.com/user-attachments/assets/ce71a75e-fca9-49a3-9702-3636d3f6c73f" />


**ฟีเจอร์หลัก:**
- บันทึก Progress ไว้ใน `progress.txt` → รันต่อจากเดิมได้เมื่อเปิดใหม่
- ขอ confirm ตอนเริ่มว่าต้องการ normalize ชื่อไฟล์เป็นรูปแบบ `img_XXXX` หรือไม่
- รองรับ Persistent Edit mode (เปิด Edit ติดข้ามภาพ)
- คลิกขวา → ลบ Box ที่ใกล้ที่สุด

**Keyboard Shortcuts:**

| ปุ่ม | การทำงาน |
|------|----------|
| `S` | บันทึก + ไปภาพถัดไป |
| `N` หรือ `0` | ไปภาพถัดไป (ไม่บันทึก) |
| `B` | กลับภาพก่อนหน้า |
| `E` | เปิด/ปิด Edit Mode |
| `P` | เปิด/ปิด Persistent Edit (ติดทุกภาพ) |
| `C` | สลับ Class (NG ↔ OK) |
| `D` / `Z` | ลบ Box ล่าสุด |
| `+` / `=` | เพิ่มความหนา Box |
| `-` | ลดความหนา Box |
| `R` | Reset ภาพปัจจุบัน |
| `H` | แสดง Help |
| `Q` | ออก |
| คลิกขวา | ลบ Box ที่ใกล้ที่สุด |

---

### 3. `easy_to_collect_update.py` — Hailo AI Accelerator
<img width="767" height="665" alt="image" src="https://github.com/user-attachments/assets/a01edf28-cc7b-4020-8f9d-b6feced134d1" />


**วัตถุประสงค์:** เก็บภาพสำหรับ Dataset โดยใช้ผลการ Detect จาก Model ที่รันอยู่บน **Hailo AI Accelerator** มาเป็น Label อัตโนมัติ ช่วยให้เก็บ Dataset ได้เร็วขึ้น

**Hardware ที่ต้องการ:** Raspberry Pi + Hailo AI Accelerator + กล้อง

**Dependencies พิเศษ:**
- `hailo_apps` — Hailo Application Framework
- `gi` (PyGObject) + GStreamer — สำหรับ Video Pipeline

**วิธีใช้:**

```bash
python easy_to_collect_update.py
```

**Classes ที่รองรับ (7 classes):**

| Class | ความหมาย (จากโค้ด) |
|-------|---------------------|
| OK | Chip ดี |
| NG | No chip |
| GL | Glue |
| MA | Misalign |
| DF | Defect |
| OG | Glue overflow |
| class6 | (ยังไม่กำหนด) |

**โครงสร้างโฟลเดอร์ Output:**

ไฟล์จะบันทึกลงโฟลเดอร์ตาม Class เช่น:
```
dataset/
├── OK/images/   OK/labels/
├── NG/images/   NG/labels/
├── GL/images/   GL/labels/
├── MA/images/   MA/labels/
├── DF/images/   DF/labels/
├── OG/images/   OG/labels/
├── class6/images/  class6/labels/
└── P/images/    P/labels/     ← ไฟล์ที่บันทึกตอน Freeze mode
```

**Keyboard Shortcuts:**

| ปุ่ม | การทำงาน |
|------|----------|
| `F` | Freeze / กลับ Live mode |
| `S` | บันทึกภาพ + label |
| `O` | เปิด/ปิด Overlay mode (ปรับกรอบ Model) |
| `A` / `D` | เลื่อน Class ก่อนหน้า/ถัดไป |
| `0`–`6` | เลือก Class โดยตรง |
| `Z` | Undo บันทึกล่าสุด |
| `Q` | ออก |
| `1`/`2`/`3`/`4`/`5`/`6`/`7`/`8` | ปรับกรอบ Overlay (ขยาย/หดแต่ละด้าน) |
| คลิกซ้ายบน Box | สลับ Class ของ Box นั้น |

**ค่าสำคัญที่แก้ในโค้ด:**

```python
"DATASET_DIR": "/home/pi/hailo-apps/hailo_apps/python/mcphase3/dataset"
"CONF_THRES": 0.20   # Confidence threshold ของ Model
```

---

### 4. `dataset_tools.py` — เครื่องมือจัดการ Dataset
<img width="516" height="273" alt="image" src="https://github.com/user-attachments/assets/7f488415-7c2a-4d9b-a4ce-446891d0cfaa" />

**วัตถุประสงค์:** เครื่องมือ CLI แบบ Interactive สำหรับงาน Dataset Management ครบวงจร

**วิธีใช้:**

```bash
python dataset_tools.py
```

จะแสดง Menu ให้เลือก:

```
==============================
YOLO DATASET TOOL
==============================
1) Merge Multiple Datasets
2) Relabel Classes
3) Remove Specific Class
4) Validate Dataset
5) Show Statistics
6) Backup Dataset
7) Split Train/Val/Test
0) Exit
```

**รายละเอียดแต่ละฟีเจอร์:**

**1) Merge Multiple Datasets**  
ค้นหาโฟลเดอร์ย่อยที่มี `images/` และ `labels/` ใน root ที่กำหนด แล้ว Merge รวมกัน พร้อม rename ไฟล์เป็น sequential (`img_00001`, `img_00002`, …)  
สร้าง `merge_manifest.csv` เก็บ log ว่าแต่ละไฟล์มาจากไหน

**2) Relabel Classes**  
เปลี่ยน class ID ใน label files ทั้งโฟลเดอร์ตาม mapping ที่ระบุ  
รูปแบบ mapping: `5:6` หรือ `5=6,2=0` หรือ `5->6;1->3`

**3) Remove Specific Class**  
ลบ annotation ของ class ID ที่ระบุออกจาก label files ทั้งหมด

**4) Validate Dataset**  
ตรวจสอบความสมบูรณ์:
- นับจำนวนภาพ / label
- หา Missing labels (มีภาพแต่ไม่มี label)
- หา Missing images (มี label แต่ไม่มีภาพ)
- นับ Empty labels
- แสดง Class Distribution

**5) Show Statistics**  
เหมือน Validate แต่เน้นแสดงสถิติ

**6) Backup Dataset**  
Copy ทั้งโฟลเดอร์ไปสำรองไว้ พร้อม timestamp เช่น `dataset_backup_20250101_120000`

**7) Split Train/Val/Test**  
แบ่ง Dataset ออกเป็น 3 ส่วน (ค่า default: Train 80%, Val 10%, Test 10%)

---

### 5. `split_dataset.py` — แบ่งข้อมูล Train/Val
<img width="702" height="104" alt="image" src="https://github.com/user-attachments/assets/3c90b89a-2206-4eb0-8453-6287e944fc07" />


**วัตถุประสงค์:** Script เบา ๆ สำหรับแบ่ง Dataset เป็น Train และ Val อย่างรวดเร็ว

**วิธีใช้:**

```bash
python split_dataset.py
```

> ⚠️ **ต้องแก้ Path ในโค้ดก่อนใช้:**
> ```python
> SOURCE_IMAGES = "dataset_raw/images"  # โฟลเดอร์ภาพต้นทาง
> SOURCE_LABELS = "dataset_raw/labels"  # โฟลเดอร์ label ต้นทาง
> OUTPUT_DIR = "dataset"                # โฟลเดอร์ปลายทาง
> TRAIN_RATIO = 0.8                     # สัดส่วน Train (80%)
> RANDOM_SEED = 42                      # Seed สำหรับความ reproducible
> ```

**Output structure:**

```
dataset/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

---

## รูปแบบ Dataset (YOLO Format)

Label ทุกไฟล์เป็น `.txt` หนึ่งบรรทัดต่อหนึ่ง Bounding Box:

```
<class_id> <x_center> <y_center> <width> <height>
```

- ค่าทุกตัวใช้ **Normalized** (0.0 – 1.0 เทียบกับขนาดภาพ)
- ถ้าภาพไม่มี object ไฟล์ `.txt` จะว่างเปล่า (ไม่ใช่ไม่มีไฟล์)

**ตัวอย่าง:**

```
0 0.512344 0.473211 0.123456 0.234567
1 0.234500 0.678900 0.100000 0.150000
```

---

## โครงสร้างโฟลเดอร์ Output

**สำหรับ Capture_Labeling_Tool.py และ Image_Labeling_Tool.py:**

```
dataset/
├── images/
│   ├── img_0000.jpg
│   ├── img_0001.jpg
│   └── ...
└── labels/
    ├── img_0000.txt
    ├── img_0001.txt
    └── ...
```

**หลัง split_dataset.py / dataset_tools.py split:**

```
dataset/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

---

## Dependencies

### สคริปต์ทั่วไป (Capture_Labeling_Tool.py, Image_Labeling_Tool.py, split_dataset.py, dataset_tools.py)

```bash
pip install opencv-python numpy
```

### Capture_Labeling_Tool.py (เพิ่มเติม — Raspberry Pi เท่านั้น)

```bash
pip install picamera2
```

### easy_to_collect_update.py (เพิ่มเติม — Hailo เท่านั้น)

ต้องติดตั้ง Hailo SDK และ `hailo_apps` ตาม documentation ของ Hailo:
- `hailo` Python package
- `hailo_apps` (Hailo Application Framework)
- `gi` (PyGObject) + GStreamer

---

## ข้อควรระวัง

- **Image_Labeling_Tool.py** มี Path แบบ Hardcode เป็น Windows (`D:\March\...`) → ต้องแก้ก่อนใช้
- **split_dataset.py** มี Path แบบ Relative Hardcode → แก้ใน `SOURCE_IMAGES`, `SOURCE_LABELS`, `OUTPUT_DIR`
- **easy_to_collect_update.py** ใช้ 7 classes แต่สคริปต์อื่นใช้แค่ 2 classes (NG/OK) → ถ้าจะ Merge Dataset ให้ตรวจสอบ class mapping ก่อน
- ควรใช้ **dataset_tools.py ฟีเจอร์ Validate** ก่อนนำ Dataset ไปเทรนทุกครั้ง
- **easy_to_collect_update.py** บันทึกไฟล์ในโฟลเดอร์ `P/` เมื่ออยู่ใน Freeze mode (ไม่ใช่โฟลเดอร์ class ปัจจุบัน) — ข้อสังเกตจากโค้ด: `FREEZE_FOLDER: "P"`

---

## Workflow แนะนำ

```
1. เก็บภาพ
   ├── Raspberry Pi           → Capture_Labeling_Tool.py
   ├── Hailo AI Accelerator   → easy_to_collect_update.py
   └── มีภาพอยู่แล้ว          → Image_Labeling_Tool.py

2. จัดการ Dataset
   └── dataset_tools.py
       ├── Merge (รวม dataset หลายชุด)
       ├── Validate (ตรวจสอบความสมบูรณ์)
       └── Backup (สำรองข้อมูล)

3. แบ่ง Train/Val
   ├── split_dataset.py         (Train/Val 80/20)
   └── dataset_tools.py → (7)   (Train/Val/Test)

4. นำ Dataset ไปเทรน YOLO Model
```
