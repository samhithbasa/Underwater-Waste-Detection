"""
Train YOLO11 on the Underwater Waste dataset.
4 classes: bottle, can, plastic-bag, waste
1521 training images
"""
from ultralytics import YOLO
import torch
import os, shutil

DATA_YAML = os.path.abspath('dataset_yolov11/data.yaml')
BASE_MODEL = 'yolo11n.pt'

def train():
    print("=" * 55)
    print(" UNDERWATER WASTE DETECTION — YOLO11 TRAINING")
    print("=" * 55)
    print(f"  Dataset : {DATA_YAML}")
    print(f"  Device  : {'GPU (CUDA)' if torch.cuda.is_available() else 'CPU'}")
    print("=" * 55)

    model = YOLO(BASE_MODEL)

    results = model.train(
        data=DATA_YAML,
        epochs=30,
        batch=2,
        imgsz=416,
        patience=5,
        workers=2,
        project='results/training',
        name='waste_v3',
        exist_ok=True,
        verbose=True,
        plots=True,
        device='cpu',
    )

    # Copy best weights to a known path so the backend picks it up automatically
    best_w = 'results/training/waste_v3/weights/best.pt'
    if os.path.exists(best_w):
        shutil.copy(best_w, 'results/best_model.pt')
        print("\n✅ Best model saved to: results/best_model.pt")
        print("   Restart app.py to use the new trained model.")
    else:
        print("\n⚠️ Training finished but best.pt not found.")

if __name__ == '__main__':
    train()
