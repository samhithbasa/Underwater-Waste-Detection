import os
import yaml
import json
from pathlib import Path

def check_dataset():
    """Check if dataset structure is correct"""
    print("Checking dataset structure...")
    
    paths_to_check = [
        'dataset_yolov10/images/train',
        'dataset_yolov10/images/val',
        'dataset_yolov10/labels/train',
        'dataset_yolov10/labels/val'
    ]
    
    for path in paths_to_check:
        if os.path.exists(path):
            files = list(Path(path).glob('*'))
            print(f"✓ {path}: {len(files)} files")
        else:
            print(f"✗ {path}: NOT FOUND")
    
    # Check train.yaml
    if os.path.exists('train.yaml'):
        print("\n✓ train.yaml found")
        with open('train.yaml', 'r') as f:
            config = yaml.safe_load(f)
            print(f"  Classes: {config.get('nc', 'Not specified')}")
            print(f"  Class names: {config.get('names', 'Not specified')}")
    else:
        print("\n✗ train.yaml NOT FOUND")

def get_device_info():
    """Get GPU/CPU information"""
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    return device

def clean_previous_results():
    """Clean previous results but keep dataset"""
    import shutil
    if os.path.exists('results'):
        shutil.rmtree('results')
        print("Cleaned previous results")
    os.makedirs('results', exist_ok=True)
