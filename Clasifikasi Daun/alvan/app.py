from flask import Flask, render_template, request
import numpy as np
from PIL import Image
import io
import joblib

app = Flask(__name__)

# Muat model dan label encoder
model = joblib.load('D:\\02. Pembelajaran Perkuliahan\\02. Pembelajaran Mesin\\Clasifikasi Daun\\alvan\\model\\KNN.joblib')
label_encoder = joblib.load('D:\\02. Pembelajaran Perkuliahan\\02. Pembelajaran Mesin\\Clasifikasi Daun\\alvan\\model\\label_encoder.joblib')

def preprocess_image(image):
    """
    Fungsi untuk memproses gambar: mengubah ke grayscale, meresize ke 16x12,
    dan meratakannya menjadi satu dimensi.
    """
    image = image.convert("L")  # Konversi gambar ke grayscale
    image = image.resize((16, 12))  # Ubah ukuran gambar menjadi 16x12
    img_array = np.array(image).flatten()  # Ratakan array gambar
    return img_array

@app.route("/", methods=["GET", "POST"])
def index():
    predicted_label = None  # Inisialisasi variabel dengan nilai None
    
    if request.method == "POST":
        # Dapatkan gambar dari form
        file = request.files.get("leaf_image")
        
        if file:
            try:
                # Buka gambar dan lakukan preprocess
                image = Image.open(io.BytesIO(file.read()))
                features = preprocess_image(image).reshape(1, -1)

                # Prediksi label
                prediction = model.predict(features)
                predicted_label = label_encoder.inverse_transform(prediction)[0]

            except Exception as e:
                predicted_label = f"Error: {e}"
        else:
            predicted_label = "No image uploaded"

    return render_template("index.html", prediction=predicted_label)

if __name__ == "__main__":
    app.run(debug=True)
