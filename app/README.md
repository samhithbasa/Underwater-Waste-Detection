# 🌊 Underwater Waste Vision

**Multi-Model Comparative Analysis for Underwater Waste Detection**

Underwater Waste Vision is a cutting-edge application designed to detect and classify waste in underwater environments using state-of-the-art computer vision models. It provides a comparative analysis interface allowing users to test and compare results across different YOLO versions.

---

## 📱 Download Application

You can download the compiled Android application (APK) directly from the repository using the link below:

[📥 **Download app-debug.apk**](https://raw.githubusercontent.com/samhithbasa/Underwater-Waste-Detection/main/app/android/app/build/outputs/apk/debug/app-debug.apk)

---

## 🚀 Key Features

- **Multi-Model Support**: Compare detections from **YOLOv9**, **YOLOv10**, and **YOLOv11**.
- **Real-time Processing**: Fast inference via a Flask backend.
- **Interactive UI**: Clean, glassmorphic interface built with React and Vite.
- **Mobile Ready**: Packaged as an Android app using Capacitor.
- **Detailed Analytics**: Returns bounding boxes, confidence scores, and dimensions for detected objects.

---

## 🛠️ Tech Stack

### Frontend
- **Framework**: [React 19](https://react.dev/)
- **Build Tool**: [Vite 7](https://vitejs.dev/)
- **Mobile Bridge**: [Capacitor 8](https://capacitorjs.com/) (Android)
- **Styling**: Vanilla CSS with premium glassmorphism effects.

### Backend
- **Framework**: [Flask](https://flask.palletsprojects.com/)
- **AI/ML Framework**: [Ultralytics YOLO](https://docs.ultralytics.com/)
- **Computer Vision**: [OpenCV](https://opencv.org/)
- **Deep Learning**: [PyTorch](https://pytorch.org/)
- **Deployment**: [Hugging Face Spaces](https://huggingface.co/spaces) / Docker

### AI Models
- **YOLOv9** (`yolov9c.pt` / `best_y9.pt`)
- **YOLOv10** (`yolov10s.pt` / `best_y10.pt`)
- **YOLOv11** (`yolo11n.pt` / `best_y11.pt`)

---

## 📁 Project Structure

```text
├── app/                  # Frontend React application
│   ├── src/              # React source code
│   ├── public/           # Static assets (logo, etc.)
│   └── android/          # Capacitor Android project
├── app.py                # Flask backend server
├── Dockerfile            # Container configuration
├── requirements.txt      # Python dependencies
└── scripts/              # Model training and export scripts
```

## ⚙️ Setup & Installation

### Prerequisites
- Node.js (for frontend)
- Python 3.10+ (for backend)
- Android Studio (for mobile builds)

### Frontend Setup
1. Navigate to the `app` directory:
   ```bash
   cd app
   ```
2. Install dependencies:
   ```bash
   npm install
   ```
3. Run the development server:
   ```bash
   npm run dev
   ```

### Backend Setup
1. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the Flask server:
   ```bash
   python app.py
   ```

---

## 📄 License
This project is part of the **AI for Planet** initiative.
