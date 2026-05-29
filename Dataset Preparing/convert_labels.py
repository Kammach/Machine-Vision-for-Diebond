import os
from pathlib import Path

def convert_labels(root_folder):
    """
    แปลงคลาสในโฟลเดอร์ labels ทั้งหมด
    - โฟลเดอร์ OK  → class 1
    - โฟลเดอร์ NG  → class 0
    """
    
    # กำหนด path หลัก
    base_path = Path(root_folder)
    
    # โฟลเดอร์ที่ต้องการแปลง
    folders = {
        "OK": 1,   # class 1 = OK
        "NG": 0    # class 0 = NG
    }
    
    total_files = 0
    converted = 0
    
    print("🚀 เริ่มแปลง Label Files...\n")
    
    for folder_name, new_class in folders.items():
        label_dir = base_path / folder_name / "labels"
        
        if not label_dir.exists():
            print(f"⚠️ ไม่พบโฟลเดอร์: {label_dir}")
            continue
            
        # หาโฟลเดอร์ train, val, test อัตโนมัติ
        subfolders = [d for d in label_dir.iterdir() if d.is_dir()]
        
        for sub in subfolders:
            print(f"📁 กำลังแปลง: {folder_name}/{sub.name} → Class {new_class}")
            
            txt_files = list(sub.glob("*.txt"))
            
            for txt_file in txt_files:
                total_files += 1
                lines = []
                changed = False
                
                with open(txt_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        
                        parts = line.split()
                        if len(parts) >= 5:
                            # แปลงคลาสเป็นคลาสใหม่
                            old_class = parts[0]
                            parts[0] = str(new_class)
                            new_line = " ".join(parts)
                            lines.append(new_line)
                            
                            if old_class != str(new_class):
                                changed = True
                        else:
                            lines.append(line)
                
                # เขียนไฟล์ใหม่
                with open(txt_file, "w", encoding="utf-8") as f:
                    f.write("\n".join(lines) + "\n")
                
                if changed:
                    converted += 1
                    
            print(f"   ✓ แปลงสำเร็จ {len(txt_files)} ไฟล์")
    
    print("\n" + "="*60)
    print("✅ แปลง Label เสร็จสิ้น!")
    print(f"   ไฟล์ทั้งหมด     : {total_files}")
    print(f"   ไฟล์ที่เปลี่ยนคลาส : {converted}")
    print("="*60)


# ==========================
# ใช้งาน
# ==========================

if __name__ == "__main__":
    # <<< แก้ path ตรงนี้ >>>
    DATASET_PATH = r"D:\March\MCphase3\chip\dataset_split\dataset"
    
    convert_labels(DATASET_PATH)
    
    input("\nกด Enter เพื่อปิด...")