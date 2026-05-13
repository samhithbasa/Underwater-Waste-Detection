"""
Fine-tune YOLO11 on the Underwater Waste dataset for better results.

Key improvements over train_model.py:
  - Starts from existing best weights (transfer learning)
  - Uses a larger model (yolo11s) if no best model exists yet
  - Higher resolution (640px) for better small-object detection
  - More epochs with cosine LR schedule
  - Targeted augmentation for underwater imagery
  - Auto GPU/CPU detection
  - Per-class mAP reporting after training
"""

import os
import shutil
import torch
from ultralytics import YOLO

# ── Choose Model ──────────────────────────────────────────────────────────────
# Options: 'yolov9c', 'yolov10s', 'yolo11s'
MODEL_TO_TRAIN = 'yolo11s'  # <-- Change this to 'yolov9c' or 'yolov10s' to train others

ROOT       = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_YAML  = os.path.join(ROOT, 'dataset_yolov11', 'data.yaml')

# Map the model choice to its local weights path
MODEL_MAP = {
    'yolov9c':  os.path.join(ROOT, 'results', 'best_y9.pt'),
    'yolov10s': os.path.join(ROOT, 'results', 'best_y10.pt'),
    'yolo11s':  os.path.join(ROOT, 'results', 'best_y11.pt')
}

BEST_MODEL = MODEL_MAP[MODEL_TO_TRAIN]
START_WEIGHTS = BEST_MODEL if os.path.exists(BEST_MODEL) else f"{MODEL_TO_TRAIN}.pt"

DEVICE = 0 if torch.cuda.is_available() else 'cpu'

# ── Hyper-parameters ───────────────────────────────────────────────────────────
EPOCHS      = 50          # Reduced for faster training on CPU
BATCH       = 4           # Increase if you have ≥8 GB VRAM / RAM
IMGSZ       = 320        # 640 is YOLO's native resolution; better for small debris
PATIENCE    = 15          # Early-stop if no improvement for 15 epochs
LR0         = 0.005       # Initial learning rate
LRF         = 0.01        # Final LR = LR0 * LRF (cosine decay)
WARMUP_EP   = 3           # Warmup epochs
OPTIMIZER   = 'AdamW'     # Adam with weight decay — more stable than SGD

# ── Augmentation (tuned for underwater imagery) ───────────────────────────────
AUGMENT_ARGS = dict(
    hsv_h      = 0.02,   # small hue shift (water colour varies)
    hsv_s      = 0.5,    # saturation (murky vs clear water)
    hsv_v      = 0.4,    # brightness (depth / lighting)
    degrees    = 10.0,   # rotation (objects tumble)
    translate  = 0.1,    # translation
    scale      = 0.6,    # zoom in/out
    shear      = 2.0,    # slight shear
    perspective= 0.0,    # keep off — distorts underwater scenes
    flipud     = 0.3,    # objects can appear upside-down underwater
    fliplr     = 0.5,
    mosaic     = 0.8,    # mosaic augmentation (helps small objects)
    mixup      = 0.1,    # mild mixup
    copy_paste = 0.1,    # copy-paste for rare classes
)


def train():
    print("=" * 60)
    print("  UNDERWATER WASTE DETECTION — YOLO11 FINE-TUNING")
    print("=" * 60)
    print(f"  Start weights : {START_WEIGHTS}")
    print(f"  Dataset       : {DATA_YAML}")
    print(f"  Device        : {'GPU (CUDA)' if DEVICE == 0 else 'CPU'}")
    print(f"  Image size    : {IMGSZ}  |  Epochs: {EPOCHS}  |  Batch: {BATCH}")
    print("=" * 60)

    model = YOLO(START_WEIGHTS)

    results = model.train(
        data       = DATA_YAML,
        epochs     = EPOCHS,
        batch      = BATCH,
        imgsz      = IMGSZ,
        patience   = PATIENCE,
        device     = DEVICE,
        optimizer  = OPTIMIZER,
        lr0        = LR0,
        lrf        = LRF,
        warmup_epochs = WARMUP_EP,
        cos_lr     = True,       # cosine annealing schedule
        workers    = 4,
        project    = os.path.join(ROOT, 'results', f'training_{MODEL_TO_TRAIN}'),
        name       = 'waste_finetune',
        exist_ok   = True,
        verbose    = True,
        plots      = True,
        val        = True,       # validate after every epoch
        save       = True,
        save_period= 10,         # checkpoint every 10 epochs
        **AUGMENT_ARGS,
    )

    # ── Save best weights ──────────────────────────────────────────────────────
    best_w = os.path.join(ROOT, 'results', f'training_{MODEL_TO_TRAIN}', 'waste_finetune', 'weights', 'best.pt')
    if os.path.exists(best_w):
        shutil.copy(best_w, BEST_MODEL)
        print(f"\n✅  Best {MODEL_TO_TRAIN} model saved → {BEST_MODEL}")
        print("    Restart app.py to serve the updated model.\n")
    else:
        print("\n⚠️  Training finished but best.pt not found — check the run folder.\n")

    # ── Per-class mAP report ───────────────────────────────────────────────────
    print("Running validation on the best checkpoint …")
    best_model = YOLO(BEST_MODEL)
    val_results = best_model.val(data=DATA_YAML, imgsz=IMGSZ, device=DEVICE, verbose=True)
    print("\n📊  Validation Summary:")
    print(f"   mAP50      : {val_results.box.map50:.4f}")
    print(f"   mAP50-95   : {val_results.box.map:.4f}")
    try:
        names = best_model.names
        for i, ap in enumerate(val_results.box.ap50):
            print(f"   {names[i]:<12}: AP50 = {ap:.4f}")
    except Exception:
        pass


if __name__ == '__main__':
    train()
