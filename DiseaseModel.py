import google.generativeai as genai
from dotenv import dotenv_values
from tensorflow.keras.utils import load_img, img_to_array  # Updated import path
import numpy as np
import tensorflow as tf
import os 

GEMINI_API_KEY = dotenv_values(".env").get("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("models/gemini-2.0-flash")

MODEL_PATH = os.path.abspath("models")


DISEASE_PREDICTION_MODEL = tf.keras.models.load_model(os.path.join( MODEL_PATH, 'trained_plant_disease_model.keras'), compile=False)

with open("labels.csv", 'r') as f:
    labels = f.readlines()
# print(labels)
class_name = labels[1:]

def predict():
    image = load_img("temp.jpg", target_size=(128, 128))
    input_arr = img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch.
    predictions = DISEASE_PREDICTION_MODEL.predict(input_arr)
    result_index = np.argmax(predictions)
    model_prediction = class_name[result_index]

    # Pre text context: interact with Gemini to fetch the reason and solution for the detected disease
    prompt = f"""
        The plant disease detected is: {model_prediction}.
        Please provide a very short explanation (2-3 sentences) about why this disease occurs, including the underlying factors.
        Additionally, offer a brief suggestion on how to manage or treat this disease with appropriate safety precautions.
        Include context for a disease solution, such as recommending a specific fertilizer or outlining a process to cure the disease.
        """
    response = gemini_model.generate_content(prompt)
    disease_explanation = response.text

    data = {
            "predicted": model_prediction,
            "explanation": disease_explanation.strip()
        }

    return data

