from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image
import io
import os
import uuid
import shutil
from pathlib import Path

app = Flask(__name__)

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "https://frontend-acne-detection-web-zeta.vercel.app"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response

# CORS hanya untuk domain frontend
CORS(app, origins=["https://frontend-acne-detection-web-zeta.vercel.app"])

# Tempat menyimpan hasil prediksi (dalam folder public untuk bisa diakses Vercel)
STATIC_FOLDER = os.path.join(app.root_path, 'static')
UPLOAD_FOLDER = os.path.join(STATIC_FOLDER, 'images')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model YOLO
model = YOLO("best.pt")  # Pastikan 'best.pt' disimpan di root project atau folder yang sesuai

def hitung_keparahan(jumlah):
    if jumlah <= 5:
        return "Ringan"
    elif jumlah <= 15:
        return "Sedang"
    else:
        return "Berat"

@app.route('/')
def home():
    return 'Hello, World!'

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    image = Image.open(io.BytesIO(file.read()))

    # Simpan input image
    image_filename = f"{uuid.uuid4().hex}.jpg"
    input_path = os.path.join(UPLOAD_FOLDER, image_filename)
    image.save(input_path)

    # Jalankan YOLO
    results = model.predict(source=input_path, save=True, conf=0.2)

    # Hitung jumlah jerawat
    jumlah_jerawat = len(results[0].boxes) if results and results[0].boxes is not None else 0
    tingkat_keparahan = hitung_keparahan(jumlah_jerawat)
    analisa_text = f"Terdeteksi {jumlah_jerawat} jerawat. Tingkat keparahan dikategorikan sebagai '{tingkat_keparahan}'."

    # Ambil gambar hasil prediksi
    result_dir = Path(results[0].save_dir)
    result_img_path = result_dir / image_filename

    # Salin ke folder statis
    final_filename = f"pred_{image_filename}"
    final_path = os.path.join(UPLOAD_FOLDER, final_filename)
    shutil.copy(result_img_path, final_path)

    # Buat URL publik
    image_url = f"https://api-acne-detection-model.vercel.app/static/images/{final_filename}"

    return jsonify({
        "image_url": image_url,
        "jumlah_jerawat": jumlah_jerawat,
        "tingkat_keparahan": tingkat_keparahan,
        "analisa": analisa_text
    })

@app.route('/static/images/<path:filename>')
def serve_image(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)
