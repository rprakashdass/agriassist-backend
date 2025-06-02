from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
import os 
from dotenv import dotenv_values
import joblib
from sklearn.preprocessing import OneHotEncoder
from WeatherAnalysis import WeatherAnalyzer
import google.generativeai as genai
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import io
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array  
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from starlette.responses import Response
from fastapi import FastAPI, HTTPException
import chromadb
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import os
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from dotenv import dotenv_values
from pydantic import BaseModel

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

WEATHER_API = dotenv_values(".env").get("WEATHER_API_KEY")
CHAT_BOT_API = dotenv_values(".env").get("GEMINI_API_KEY")
genai.configure(api_key=CHAT_BOT_API)
gemini_model = genai.GenerativeModel("models/gemini-2.0-flash")
# Load the embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
# Define the directory where pest images are stored
PEST_IMAGE_DIR = "./pest_dataset"
# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection(name="pest_data")

MODEL_PATH = os.path.abspath("models")

PEST_RISK_SUGGESSTION_MODEL = None
DISEASE_PREDICTION_MODEL = None

city = "Coimbatore"
    
analyzer = WeatherAnalyzer(WEATHER_API, city)

pest_data_processed = pd.read_csv(os.path.join(os.path.abspath("Datasets"), "pest_.csv"))

plant_feature_names = None
class_name = None

def preprocessing():
    global plant_feature_names, class_name

    def extract_humidity_avg(humidity_range):
        low, high = map(int, humidity_range.split('-'))
        return (low + high) / 2
    
    # Create a mapping dictionary for rain values to percentages
    rain_mapping = {
        'Low': 20,       # Low rain -> 20% chance
        'Moderate': 50,  # Moderate rain -> 50% chance
        'High': 80       # High rain -> 80% chance
    }

    #  a new column with numerical percentage values
    pest_data_processed['rain_percentage'] = pest_data_processed['rain'].map(rain_mapping)
    
    pest_data_processed['humidity_avg'] = pest_data_processed['humidity'].apply(extract_humidity_avg)

    # Create features (X) and target (y)
    X = pest_data_processed[['plant', 'avg_temp', 'humidity_avg', 'rain_percentage']]
    y = pest_data_processed['insect_name']

    # One-hot encode the plant names
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    plant_encoded = encoder.fit_transform(X[['plant']])
    plant_feature_names = encoder.get_feature_names_out(['plant'])

    with open("labels.csv",'r') as f:
        labels = f.readlines()
    print(labels)
    class_name = labels[1:] # validation_set.class_names

preprocessing()

def load_models():
    # Load the trained pest prediction model
    global PEST_RISK_SUGGESSTION_MODEL, DISEASE_PREDICTION_MODEL
    try:
        # Load the model from file
        print(os.path.join(MODEL_PATH,'pest_prediction_model.joblib'))
        model = joblib.load(os.path.join(MODEL_PATH,'pest_prediction_model.joblib'))
        print("Pest prediction model loaded successfully")
        
        # Display model information
        print(f"Model type: {type(model).__name__}")
        # print(f"Number of trees in forest: {model.n_estimators}")
        print(f"Number of insect classes: {len(model.classes_)}")
        print(f"Insect classes: {', '.join(model.classes_[:5])}...")
        
        PEST_RISK_SUGGESSTION_MODEL = model
        
    except FileNotFoundError:
        print("Error: Pest risk suggestion model file not found.")
    except Exception as e:
        print(f"Error loading model: {str(e)}")

    try:
        DISEASE_PREDICTION_MODEL = tf.keras.models.load_model(os.path.join(MODEL_PATH,'trained_plant_disease_model.keras'))
    
    except FileNotFoundError:
        print("Error: Plant disease prediction model file not found.")
    
    except Exception as e:
        print(f"Error loading model: {str(e)}")

load_models()

def setup_chat_bot():
    """Configure and return the Gemini model"""
    genai.configure(api_key=CHAT_BOT_API)
    # For free API key, use the standard Gemini model
    # Use the Gemini 2.0 Flash model
    model_name = 'models/gemini-2.0-flash'
    print(f"Using model: {model_name}")
    return genai.GenerativeModel(model_name)

def process_user_query(query):
    
    model = setup_chat_bot()
    
    prompt = f"""
        USER QUERY:
        {query}
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error communicating with Gemini API: {str(e)}"

# Function to predict insects for a given plant based on current weather
def predict_pest_risk(plant_name, current_weather):
    model = PEST_RISK_SUGGESSTION_MODEL
    print(current_weather)
    # Extract relevant weather info
    temp = current_weather.get('temp', 0)
    humidity = current_weather.get('humidity', 0)
    rain_chance = current_weather.get('rain_chance', 0)
    
    # Prepare input for prediction
    plant_input = np.zeros((1, len(plant_feature_names)))
    try:
        plant_idx = np.where(plant_feature_names == f'plant_{plant_name}')[0][0]
        plant_input[0, plant_idx] = 1
    except:
        print(f"Warning: Plant '{plant_name}' not found in training data")
        
    weather_input = np.array([[temp, humidity, rain_chance]])
    input_data = np.hstack([plant_input, weather_input])
    
    # Get probabilities for each insect
    probas = model.predict_proba(input_data)[0]
    
    # Get top 3 insects with highest probabilities
    top_indices = probas.argsort()[-3:][::-1]
    top_insects = [(model.classes_[i], probas[i]) for i in top_indices]
    
    # Get damage descriptions for these insects
    results = []
    for insect, prob in top_insects:
        damage = pest_data_processed[pest_data_processed['insect_name'] == insect]['damage'].values[0]
        results.append({
            'insect': insect,
            'probability': prob * 100,  # Convert to percentage
            'damage': damage
        })
    print(results)
    return results

def predict():
    image = load_img("temp.jpg", target_size=(128, 128))
    input_arr = img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch.
    predictions = DISEASE_PREDICTION_MODEL.predict(input_arr)
    result_index = np.argmax(predictions)
    model_prediction = class_name[result_index]

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

@app.get("/warnings")
def warnings():
    warnings = {
        "Warning":False
    }
    if not PEST_RISK_SUGGESSTION_MODEL:
        warnings["Warning"]="Pest risk suggestion model not loaded!"
    elif not DISEASE_PREDICTION_MODEL:
        warnings["Warning"]="Disease prediction model not loaded!"
    
    return warnings

@app.get("/fetch-current-weather-data")
def fetch_current_weather_data():
    return analyzer.fetch_current_weather()

@app.get("/chat/")
def chat(query: str):
    return {
        "Response": process_user_query(query)
    }

@app.get("/predict/{plant_name}")
def predict_pest(plant_name: str):
    response = predict_pest_risk(plant_name, fetch_current_weather_data())
    return response


@app.post("/upload/plant-image")
async def upload_image(file: UploadFile = File(...)):
    try:
        # Read image bytes and create PIL image
        image_bytes = await file.read()
        pil_image = Image.open(io.BytesIO(image_bytes))
        # Optionally, process or convert the PIL image if needed
        
        # Save the image as 'temp.jpg' for disease prediction processing
        pil_image.save("temp.jpg")
        
        # Call the disease prediction function from DiseaseModel.py
        disease_result = predict()
        
        return {
            "status": "success",
            "message": "Image uploaded and processed successfully.",
            "disease_prediction": disease_result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.get("/predict-disease/plant-image")
async def predict_disease_image():
    image = tf.keras.preprocessing.image.load_img("temp.jpg",target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
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


