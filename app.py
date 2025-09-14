from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from tensorflow.keras.models import model_from_json

app = FastAPI()

# Allow CORS supaya bisa diakses dari Netlify
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load CNN model
with open("facial_expression_model_structure.json", "r") as f:
    model = model_from_json(f.read())
model.load_weights("facial_expression_model_weights.h5")

# Load Haar Cascade face detector
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Emotions sesuai training
emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # baca file gambar
    img_bytes = await file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    results = []
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi_gray = roi_gray.astype("float") / 255.0
        roi_gray = np.expand_dims(roi_gray, axis=-1)
        roi_gray = np.expand_dims(roi_gray, axis=0)

        preds = model.predict(roi_gray, verbose=0)[0]
        emotion_label = emotions[np.argmax(preds)]

        results.append({
            "box": [int(x), int(y), int(w), int(h)],
            "emotion": emotion_label
        })

    return {"results": results}
