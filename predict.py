import json
import numpy as np
import tensorflow as tf
from PIL import Image

MODEL_PATH = "saved_model/fruit_vgg16.keras"
CLASS_PATH = "saved_model/class_names.json"

import os
import urllib.request


if (not os.path.exists(MODEL_PATH)) or os.path.getsize(MODEL_PATH) < 1000000:
    os.makedirs("saved_model", exist_ok=True)

    urllib.request.urlretrieve(
        "https://github.com/zerathionvoid-ctrl/fruit-classification-vgg16/releases/download/v1.0/fruit_vgg16.keras",
        MODEL_PATH
    )

model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_PATH, "r") as f:
    class_names = json.load(f)

IMAGE_SIZE = (224, 224)

def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize(IMAGE_SIZE)

    img_array = np.array(image, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array, verbose=0)

    index = np.argmax(prediction)

    label = class_names[index]

    confidence = float(prediction[0][index]) * 100

    return label, round(confidence,2)