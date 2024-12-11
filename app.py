import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from PIL import Image
import io
import logging

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

# Load the .h5 model
model_path = "foods_model.h5"
model = tf.keras.models.load_model(model_path)

food_labels = [
    "amparan_tatak", "asam_padeh", "asem_asem_daging", "ayam_geprek",
    "ayam_goreng_tepung", "ayam_pop", "ayam_rica_rica", "ayam_tangkap",
    "babi_guling", "babi_panggang_karo", "bakmi_jawa", "bakpia", 
    "bakso", "bakwan", "barongko", "batagor", "bika_ambon", 
    "bolu_kojo", "botok_telur_asin", "bubur_ayam", "burgo",
    "cimol", "cireng", "combro", "coto_makassar", "daging_masak_hitam",
    "dawet", "dendeng_balado", "es_puter", "es_selendang_mayang",
    "gado_gado", "gudeg", "gulai", "jalangkote", "kapurung",
    "karedok", "keciput", "kerak_telor", "kolak", "kue_cubit",
    "kue_jojorong", "kue_kamir", "kue_kipo", "kue_klepon", 
    "kue_nagasari", "kue_putu", "kue_satu", "kue_serabi", 
    "kue_wajik", "kwetiau_bagan", "laksamana_mengamuk", "laksan",
    "lalampa", "lawar", "luti_gendang", "mendol", "mi_aceh",
    "nasi_goreng", "nasi_uduk", "opor_ayam", "pallu_butung", 
    "pempek", "pepes", "perkedel", "pindang_serani", "pisang_ijo",
    "pisang_molen", "pisang_peppe", "plecing_kangkung", "pukis",
    "rawon", "rendang", "rengginang", "roti_buaya", "roti_canai",
    "roti_ganjel_rel", "roti_jala", "rujak_kuah_pindang", "sate",
    "sate_buntel", "sate_padang", "seblak", "selat_solo", 
    "semur_jengkol", "sop_konro", "soto_ayam", "soto_bandung",
    "tahu_aci", "tahu_gejrot", "tahu_sumedang", "tapai_singkong",
    "tinutuan", "tumpeng"
]

CONFIDENCE_THRESHOLD = 0.5

def preprocess_image(image):
    """
    Preprocessing gambar sebelum dimasukkan ke dalam model:
    - Resize gambar sesuai ukuran input model.
    - Normalize piksel ke rentang [0, 1].
    """
    input_shape = model.input_shape[1:3]  # Ambil ukuran input dari model
    if image.size != tuple(input_shape):
        image = image.resize(input_shape)
    image = np.array(image, dtype=np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400

        file = request.files['image']
        
        try:
            image = Image.open(io.BytesIO(file.read())).convert('RGB')
        except Exception:
            return jsonify({'error': 'Invalid image format'}), 400

        processed_image = preprocess_image(image)

        predictions = model.predict(processed_image)
        confidence = float(np.max(predictions))

        if confidence < CONFIDENCE_THRESHOLD:
            return jsonify({'error': 'Low confidence', 'message': 'selected image has an error', 'confidence': confidence}), 200

        predicted_class = np.argmax(predictions)
        predicted_food_name = food_labels[predicted_class]

        response = {
            'predicted_class': int(predicted_class),
            'predicted_class_name': predicted_food_name,
            'confidence': confidence
        }

        logging.info(f"Prediction: {response}")

        return jsonify(response)

    except Exception as e:
        logging.error(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)