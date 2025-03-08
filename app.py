import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from utils import preprocess_image, decode_predictions  

app = Flask(__name__)

# Set upload folder
UPLOAD_FOLDER = "static/uploads/"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load OCR Model
ocr_model_path = "ocr_character_model.h5"
if os.path.exists(ocr_model_path):
    ocr_model = tf.keras.models.load_model(ocr_model_path)
    print("✅ OCR Model Loaded Successfully!")
    print("Expected model input shape:", ocr_model.input_shape)
else:
    ocr_model = None
    print(f"⚠️ OCR model not found at {ocr_model_path}")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return "No file part"

    file = request.files["file"]
    if file.filename == "":
        return "No selected file"

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        # Process Image
        processed_img = preprocess_image(filepath)

        # Run OCR model if available
        if ocr_model:
            predictions = ocr_model.predict(processed_img)

            # Debugging: Print the raw model output
            print("Raw Model Predictions:", predictions)

            predicted_text = decode_predictions(predictions)
        else:
            predicted_text = "OCR Model Not Found"

        return f"Predicted Text: {predicted_text}"

    return redirect(url_for("home"))

if __name__ == "__main__":
    app.run(debug=True)
