"""
Train YOLO11 on the new SeaClear Marine Debris dataset.
This dataset has high-quality localized bounding boxes which will fix the 'full-frame' issue.
"""

import os
import shutil
import torch
from ultralytics import YOLO

# ── Configuration ──────────────────────────────────────────────────────────────
ROOT       = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_YAML  = os.path.join(ROOT, 'dataset_seaclear', 'data.yaml')
BASE_MODEL = 'yolo11n.pt'  # Starting from scratch/pretrained is better for new classes
BEST_MODEL = os.path.join(ROOT, 'results', 'best_y11.pt')

DEVICE = 0 if torch.cuda.is_available() else 'cpu'

# ── Hyper-parameters ───────────────────────────────────────────────────────────
# Note: Training on 8000 images on CPU is slow. 
# We'll do 10 epochs as a 'fast' fine-tune.
EPOCHS      = 10
BATCH       = 8           # Adjusted for memory
IMGSZ       = 320         # 320 for speed, 640 for accuracy
PATIENCE    = 5

def train():
    print("=" * 60)
    print("🚀 TRAINING ON SEACLEAR DATASET (FIX FOR BOUNDING BOXES)")
    print("=" * 60)
    print(f"Dataset   : {DATA_YAML}")
    print(f"Device    : {'GPU (CUDA)' if DEVICE == 0 else 'CPU'}")
    print(f"Parameters: Epochs={EPOCHS}, ImgSz={IMGSZ}, Batch={BATCH}")
    print("=" * 60)

    # Backup the old 'broken' model just in case
    if os.path.exists(BEST_MODEL):
        backup_path = BEST_MODEL.replace('.pt', '_old_fullframe.pt')
        shutil.copy(BEST_MODEL, backup_path)
        print(f"📦 Backed up old model to: {os.path.basename(backup_path)}")

    model = YOLO(BASE_MODEL)

    results = model.train(
        data       = DATA_YAML,
        epochs     = EPOCHS,
        batch      = BATCH,
        imgsz      = IMGSZ,
        patience   = PATIENCE,
        device     = DEVICE,
        project    = os.path.join(ROOT, 'results', 'training_seaclear'),
        name       = 'y11_seaclear_fix',
        exist_ok   = True,
        verbose    = True,
        plots      = True,
        save       = True,
    )

    # ── Save best weights ──────────────────────────────────────────────────────
    best_w = os.path.join(ROOT, 'results', 'training_seaclear', 'y11_seaclear_fix', 'weights', 'best.pt')
    if os.path.exists(best_w):
        shutil.copy(best_w, BEST_MODEL)
        print(f"\n✅ SUCCESS: New model with accurate bounding boxes saved → {BEST_MODEL}")
        print("   Restart the Flask app to use the new model.")
    else:
        print("\n❌ Error: Training finished but best weights were not found.")

if __name__ == '__main__':
    train()
