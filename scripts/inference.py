import os
import cv2
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np

def setup_inference():
    """Setup for inference"""
    os.makedirs('results/predictions', exist_ok=True)
    
    # Check if model exists
    model_path = 'results/best_model.pt'
    if not os.path.exists(model_path):
        model_path = 'yolo11n.pt'
    
    return model_path

def run_inference():
    """Run inference on test images"""
    print("\n" + "="*50)
    print("RUNNING INFERENCE")
    print("="*50)
    
    model_path = setup_inference()
    
    # Load model
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    
    # Test images path
    test_path = 'dataset_yolov10/images/test'
    if not os.path.exists(test_path):
        test_path = 'dataset_yolov10/images/val'
    
    # Get test images
    if os.path.exists(test_path):
        test_images = list(Path(test_path).glob('*.jpg')) + \
                     list(Path(test_path).glob('*.png')) + \
                     list(Path(test_path).glob('*.jpeg'))
        
        if test_images:
            print(f"Found {len(test_images)} test images")
            
            # Run inference on first 5 images (for quick testing)
            for i, img_path in enumerate(test_images[:5]):
                print(f"\nProcessing: {img_path.name}")
                
                # Predict
                results = model.predict(
                    source=str(img_path),
                    conf=0.25,
                    iou=0.45,
                    save=True,
                    save_dir='results/predictions',
                    save_txt=True,
                    exist_ok=True
                )
                
                # Show results
                for r in results:
                    orig_h, orig_w = r.orig_shape
                    print(f"  Image Size: {orig_w}x{orig_h}")
                    
                    for box in r.boxes:
                        # Pixel dimensions
                        x1, y1, x2, y2 = box.xyxy[0]
                        w_px = x2 - x1
                        h_px = y2 - y1
                        
                        # Class name
                        cls = model.names[int(box.cls)]
                        conf = float(box.conf)
                        
                        print(f"  - [{cls}] {w_px:.1f}x{h_px:.1f} px (conf: {conf:.2f})")

                    im_array = r.plot()
                    output_path = f'results/predictions/{img_path.stem}_result.jpg'
                    cv2.imwrite(output_path, im_array)
                    print(f"  ✓ Saved visual: {output_path}")
        else:
            print("No test images found!")
    else:
        print(f"Test path not found: {test_path}")

def main():
    try:
        run_inference()
        print("\n✅ Inference completed!")
        print("📁 Predictions saved in: results/predictions/")
    except Exception as e:
        print(f"\n❌ Error during inference: {e}")

if __name__ == '__main__':
    main()