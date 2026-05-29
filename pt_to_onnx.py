import os
# os.environ["YOLO_OFFLINE"] = "True"

from ultralytics import YOLO
import onnxruntime as ort

# =========================
# CONFIG
# =========================
PT_MODEL = r"D:\March\MCphase3\chip\bestchip100.pt"
ONNX_MODEL = r"D:\March\MCphase3\chip\bestchip100.onnx"
IMG_SIZE = 640

# =========================
# STEP 1: EXPORT PT -> ONNX
# =========================
print("🚀 Exporting PyTorch (.pt) -> ONNX...")

model = YOLO(PT_MODEL)

export_path = model.export(
    format="onnx",
    opset=11,
    imgsz=IMG_SIZE,
    simplify=True,
    nms=False,
    dynamic=False,
    optimize=False  # ✅ IMPORTANT
)

# ✅ Use returned path instead of guessing
ONNX_MODEL = export_path

if not os.path.exists(ONNX_MODEL):
    raise Exception("❌ ONNX export failed!")

print("✅ ONNX export completed:", ONNX_MODEL)

# =========================
# STEP 2: VALIDATE ONNX
# =========================
print("🔍 Validating ONNX model...")

try:
    session = ort.InferenceSession(ONNX_MODEL)
    inputs = session.get_inputs()
    outputs = session.get_outputs()

    print("✅ ONNX model loaded successfully!")
    print("📥 Input:", inputs[0].name, inputs[0].shape)
    print("📤 Output:", outputs[0].name, outputs[0].shape)

except Exception as e:
    print("❌ ONNX validation failed:")
    print(e)
    exit()

print("🎉 DONE! Your ONNX is ready for Hailo 🚀")