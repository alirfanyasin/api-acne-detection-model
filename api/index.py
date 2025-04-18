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
CORS(app)

UPLOAD_FOLDER = 'results/images'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = YOLO("best.pt")

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

    # Simpan gambar input sementara
    image_filename = f"{uuid.uuid4().hex}.jpg"
    input_path = os.path.join(UPLOAD_FOLDER, image_filename)
    image.save(input_path)

    # Jalankan prediksi YOLO
    results = model.predict(source=input_path, save=True, conf=0.2)

    # Hitung jumlah deteksi dari hasil YOLO
    jumlah_jerawat = len(results[0].boxes) if results and results[0].boxes is not None else 0
    tingkat_keparahan = hitung_keparahan(jumlah_jerawat)
    analisa_text = f"Terdeteksi {jumlah_jerawat} jerawat. Tingkat keparahan dikategorikan sebagai '{tingkat_keparahan}'."

    # Ambil direktori hasil YOLO (runs/detect/predictN)
    result_dir = Path(results[0].save_dir)
    result_img_path = result_dir / image_filename

    # Pindahkan hasil ke folder results/images
    final_filename = f"pred_{image_filename}"
    final_path = os.path.join(UPLOAD_FOLDER, final_filename)
    shutil.copy(result_img_path, final_path)

    # Kirim URL dan analisa ke frontend
    return jsonify({
        "image_url": f"http://localhost:5000/results/images/{final_filename}",
        "jumlah_jerawat": jumlah_jerawat,
        "tingkat_keparahan": tingkat_keparahan,
        "analisa": analisa_text
    })

@app.route('/results/images/<path:filename>')
def serve_image(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)
