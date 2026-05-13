"""
Export fine-tuned YOLO11 model to mobile-friendly formats (TFLite, ONNX).
This is a critical step for mobile app development, allowing for on-device inference.
"""

import os
from ultralytics import YOLO

# ── Configuration ──────────────────────────────────────────────────────────────
ROOT       = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
BEST_MODEL = os.path.join(ROOT, 'results', 'best_y11.pt')
OUTPUT_DIR = os.path.join(ROOT, 'results', 'mobile_exports')

def export_for_mobile():
    print("=" * 60)
    print("📱 EXPORTING MODEL FOR MOBILE DEPLOYMENT")
    print("=" * 60)

    if not os.path.exists(BEST_MODEL):
        print(f"❌ Error: Best model not found at {BEST_MODEL}")
        print("Please run training first.")
        return

    # Load the fine-tuned model
    model = YOLO(BEST_MODEL)

    # 1. Export to ONNX (Universal format)
    print("\n📦 Exporting to ONNX ...")
    model.export(format='onnx', dynamic=True)

    # 2. Export to TFLite (Android)
    # We use int8 quantization to significantly reduce size for mobile
    print("\n📦 Exporting to TFLite (Quantized) ...")
    try:
        model.export(format='tflite', int8=True)
        print("✅ TFLite export complete.")
    except Exception as e:
        print(f"⚠️ TFLite export failed (likely missing dependencies): {e}")

    # 3. Export to CoreML (iOS)
    print("\n📦 Exporting to CoreML ...")
    try:
        model.export(format='coreml')
        print("✅ CoreML export complete.")
    except Exception as e:
        print(f"⚠️ CoreML export failed (likely missing coremltools): {e}")

    print("\n" + "=" * 60)
    print(f"🎉 MOBILE EXPORTS SAVED IN: {os.path.dirname(BEST_MODEL)}")
    print("=" * 60)

if __name__ == '__main__':
    export_for_mobile()
