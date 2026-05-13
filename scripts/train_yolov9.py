import os
import shutil
import torch
from ultralytics import YOLO

# ── Configuration ──────────────────────────────────────────────────────────────
MODEL_TYPE = 'yolov9c' 
ROOT       = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_YAML  = os.path.join(ROOT, 'dataset_seaclear', 'data.yaml')
BEST_MODEL = os.path.join(ROOT, 'results', 'best_y9.pt')
START_WEIGHTS = 'yolov9c.pt' # Reset to base to fix full-frame bias

DEVICE = 0 if torch.cuda.is_available() else 'cpu'

# ── Hyper-parameters ───────────────────────────────────────────────────────────
EPOCHS      = 50
BATCH       = 8   # YOLOv9c is larger, keep batch smaller than s models
IMGSZ       = 320
PATIENCE    = 10

def train():
    print(f"🚀 Starting YOLOv9 Final Training...")
    model = YOLO(START_WEIGHTS)

    results = model.train(
        data       = DATA_YAML,
        epochs     = EPOCHS,
        batch      = BATCH,
        imgsz      = IMGSZ,
        patience   = PATIENCE,
        device     = DEVICE,
        project    = os.path.join(ROOT, 'results', 'training_yolov9'),
        name       = 'waste_finetune',
        exist_ok   = True,
        verbose    = True,
        workers    = 2,
    )

    # Save best weights
    best_w = os.path.join(ROOT, 'results', 'training_yolov9', 'waste_finetune', 'weights', 'best.pt')
    if os.path.exists(best_w):
        shutil.copy(best_w, BEST_MODEL)
        print(f"\n✅  YOLOv9 model saved → {BEST_MODEL}")

if __name__ == '__main__':
    train()
