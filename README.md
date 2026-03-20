# Deepfake Detector 🔍

An AI-powered web application that detects deepfake / AI-generated face images using **MobileNetV2** transfer learning, served via a **FastAPI + TensorFlow** Python backend with a modern, responsive frontend.

---

## Features

- 🖼️ **Drag & Drop** image upload (or single-click to browse)
- 🖱️ **Double-click** any gallery sample to load it instantly
- 🔬 **MobileNetV2** model trained on real vs AI-generated faces
- 📊 Color-coded result cards with confidence bar (✅ Real / ❌ Fake)
- 🌙 Dark / Light mode toggle
- 📱 Fully **responsive** — desktop, tablet & mobile
- 🍔 Mobile hamburger navigation menu
- ⚡ Smooth scroll navigation

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | HTML5, CSS3 (Vanilla), JavaScript (ES6+) |
| Backend | Python, FastAPI, Uvicorn |
| ML Model | TensorFlow / Keras — MobileNetV2 |
| Image Processing | Pillow, NumPy |

---

## Project Structure

```
├── app.py                        # FastAPI backend
├── deepfake_mobilenetv2_model.h5 # Trained Keras model
├── static/
│   ├── index.html                # Frontend UI
│   ├── styles.css                # Responsive stylesheet
│   ├── script.js                 # Frontend logic
│   └── *.jpg                     # Dataset sample images
├── tfjs_model/                   # TensorFlow.js model files
├── requirements.txt
└── .gitignore
```

---

## Setup & Run

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### 2. Create a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the server
```bash
python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

### 5. Open in browser
```
http://127.0.0.1:8000
```

---

## How It Works

1. **Upload** a face image via drag-and-drop or file browser
2. Click **Analyze Image**
3. The image is sent to the FastAPI backend
4. The MobileNetV2 model predicts whether the face is **Real** or **AI-generated (Fake)**
5. The result is displayed with a confidence score and color-coded card

---

## Model

- Architecture: **MobileNetV2** (Transfer Learning)
- Input size: **224 × 224 × 3** (RGB)
- Output: Sigmoid — `score > 0.5 → Real`, `score ≤ 0.5 → Fake`
- Trained on a curated dataset of real and AI-generated faces

---

## Institution

**KIIT University**, Bhubaneswar, Odisha — AI Deepfake Detection Project © 2026
