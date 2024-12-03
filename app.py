import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from PIL import Image
import io
import logging

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

model_path = "padang_food_model.tflite"
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

food_labels = [
    "amparan_tatak", "asam_padeh", "asem_asem_daging", "ayam_geprek",
    "ayam_goreng_tepung", "ayam_pop", "ayam_rica_rica", "ayam_taliwang",
    "ayam_tangkap", "babi_guling", "babi_panggang_karo", "bakmi_jawa",
    "bakpia", "bakso", "bakwan", "barongko",
    "batagor", "bika_ambon", "bingka", "binte_biluhuta",
    "bolu_kojo", "botok_telur_asin", "brengkes_tempoyak", "bubur_ayam",
    "bubur_kampiun", "bubur_pedas", "bubur_sagu_ambon", "burgo",
    "celimpungan", "cilok", "cimol", "cireng",
    "combro", "coto_makassar", "daging_masak_hitam", "dawet",
    "dendeng_balado", "empal_gentong", "emping", "es_puter",
    "es_selendang_mayang", "gado_gado", "garang_asem", "gehu",
    "gepuk", "gudeg", "gulai", "horok_horok",
    "jadah_manten", "jagung_bose", "jaja_klepon", "jaja_wajik",
    "jalangkote", "jemblem", "kaledo", "kalio",
    "kalumpe", "kapurung", "karedok", "keciput",
    "kerak_telor", "kerupuk_basah", "kerutup_ikan", "ketoprak",
    "kolak", "kuah_pliek_u", "kue_cubit", "kue_jojorong",
    "kue_kipo", "kue_lam", "kue_muso", "kue_nagasari",
    "kue_putu", "kue_satu", "kue_serabi", "kwetiau_bagan",
    "laksamana_mengamuk", "laksan", "lalampa", "lawar",
    "lontong_kupang", "luti_gendang", "mendol", "mi_aceh",
    "mi_gomak", "mi_kangkung", "mi_kopyok", "mi_nyemek",
    "nasi_gandul", "nasi_goreng", "nasi_kebuli", "nasi_krawu",
    "nasi_liwet", "nasi_timbel", "nasi_uduk", "opor_ayam",
    "pallu_butung", "pempek", "pepes", "perkedel",
    "pindang_serani", "pisang_ijo", "pisang_molen", "pisang_peppe", 
    "plecing_kangkung", "pukis", "rawon", "rendang",
    "rengginang", "roti_buaya", "roti_canai", "roti_ganjel_rel",
    "roti_jala", "rujak", "rujak_kuah_pindang", "sate",
    "sate_buntel", "sate_padang", "seblak", "selat_solo",
    "semur", "semur_jengkol", "serabi_solo", "siomai",
    "sop_konro", "soto_ayam", "soto_bandung","soto_betawi",
    "tahu_aci", "tahu_bakso", "tahu_gejrot", "tahu_sumedang",
    "tahu_tek", "tapai_singkong", "tekwan", "tempe_penyet",
    "tinutuan", "tongseng", "tumpeng", "untir_untir",
    "wedang_ronde","wingko_babat"
]

CONFIDENCE_THRESHOLD = 0.5

def preprocess_image(image):
    """
    Preprocessing gambar sebelum dimasukkan ke dalam model:
    - Resize gambar sesuai ukuran input model.
    - Normalize piksel ke rentang [0, 1].
    """
    input_shape = input_details[0]['shape'][1:3]
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

        interpreter.set_tensor(input_details[0]['index'], processed_image)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])

        confidence = float(np.max(predictions))
        if confidence < CONFIDENCE_THRESHOLD:
            return jsonify({'error': 'Low confidence', 'confidence': confidence}), 200

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
