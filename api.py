from keras.models import load_model
from PIL import Image
import tensorflow as tf
import numpy as np
import flask
import io
from gemini_api import recommendation as gemini_recommendation
from flask_cors import CORS

def CustomBatchNormalization(**kwargs):
    return tf.keras.layers.BatchNormalization(**kwargs)

class_names = ['healthy', 'powdery', 'rust']

model = load_model('pdr.h5', compile=True, custom_objects={'CustomBatchNormalization': CustomBatchNormalization})

app = flask.Flask(__name__)
CORS(app)

@app.route("/predict", methods=["POST"])
def predict():
    data = {"success": False}
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            image_file = flask.request.files["image"]
            image_bytes = image_file.read()
            image = np.array(Image.open(io.BytesIO(image_bytes)).resize((224, 224))) / 255.0
            image = np.expand_dims(image, axis=0)

            try:
                preds = model.predict(image)
                label = np.argmax(preds)
                predicted_class = class_names[label]
                recom = gemini_recommendation(predicted_class)
                data["success"] = True
                data["prediction"] = predicted_class
                data["recommendation"] = recom

            except Exception as e:
                data["error"] = str(e)

    return flask.jsonify(data)

@app.route('/', methods=['GET'])
def home():
    return flask.jsonify('OK')

if __name__ == "__main__":
    app.run()
