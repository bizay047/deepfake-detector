import os
import io
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "deepfake_mobilenetv2_model.h5")

model = None
try:
    model = tf.keras.models.load_model(model_path)
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Root route → load frontend
@app.get("/")
async def read_index():
    return FileResponse("static/index.html")

# Health check (important for Render)
@app.get("/health")
async def health():
    return {"status": "ok"}

# Prediction API
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        if model is None:
            return JSONResponse(
                status_code=500,
                content={"error": "Model not loaded"}
            )

        # Read image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Preprocess
        image = image.resize((224, 224))
        img_array = np.array(image, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        prediction = model.predict(img_array)
        score = float(prediction[0][0])

        # Classification
        if score > 0.5:
            label = "Real"
            confidence = round(score * 100, 1)
        else:
            label = "Fake"
            confidence = round((1.0 - score) * 100, 1)

        print(f"DEBUG → score={score:.4f}, label={label}")

        return {
            "label": label,
            "confidence": confidence,
            "raw_score": round(score, 4)
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
