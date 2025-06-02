"""
AgriAssist - Plant Disease Detection API
A FastAPI application for plant disease detection and agricultural assistance
"""

# Standard library imports
import os
import tempfile
import logging
from datetime import datetime
from io import BytesIO

# Third-party imports
import numpy as np
import tensorflow as tf
import pandas as pd
import uvicorn
from PIL import Image
from dotenv import dotenv_values
import google.generativeai as genai
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import torch
import torch.nn as nn
from torchvision import models, transforms
import matplotlib.pyplot as plt

# Local imports
from WeatherAnalysis import WeatherAnalyzer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize logging system
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='app.log'
)
logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(title="Plant Disease Detection API")

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# In-memory log storage for the warnings endpoint
log_records = []

# Custom logging handler for in-memory log storage
class MemoryHandler(logging.Handler):
    def emit(self, record):
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": record.levelname,
            "message": record.getMessage()
        }
        log_records.append(log_entry)

# Configure logger with memory handler
memory_handler = MemoryHandler()
logger.addHandler(memory_handler)

# Initialize service variables
weather_analyzer = None
is_chatbot_available = False
gemini_model = None

# Load Weather API service
try:
    WEATHER_API = dotenv_values(".env").get("WEATHER_API_KEY")
    city = "Coimbatore"  # Default city
    weather_analyzer = WeatherAnalyzer(WEATHER_API, city)
    logger.info("Weather service initialized successfully")
except Exception as e:
    logger.error(f"Weather API initialization failed: {str(e)}")

# Initialize Gemini AI chatbot
try:
    GEMINI_API_KEY = dotenv_values(".env").get("GEMINI_API_KEY")
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel("models/gemini-2.0-flash")
    is_chatbot_available = True
    logger.info("Gemini AI chatbot initialized successfully")
except Exception as e:
    logger.error(f"Gemini AI initialization failed: {str(e)}")

def load_pest_model(model_path, num_classes=102):
    """
    Load the pest detection model

    Args:
        model_path: Path to the model file
        num_classes: Number of pest classes

    Returns:
        Loaded PyTorch model or None if loading fails
    """
    try:
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()
        logger.info("Pest detection model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading pest model: {str(e)}")
        return None

# Load disease prediction model and class labels
try:
    MODEL_DIR = "models"
    
    # Load plant disease model
    plant_model_path = os.path.join(MODEL_DIR, 'trained_plant_disease_model.keras')
    DISEASE_PREDICTION_MODEL = tf.keras.models.load_model(plant_model_path)
    
    plant_disease_df = pd.read_csv('Labels/labels.csv')
    class_name = list(plant_disease_df['Plant_Disease_Labels'])
    logger.info("Plant disease model loaded successfully")
    
    # Load pest detection model
    pest_model_path = os.path.join(MODEL_DIR, 'resnet50_ip102.pth')
    pest_classes_file = os.path.join('Labels', 'classes.txt')
    
    try:
        with open(pest_classes_file, 'r') as f:
            PEST_CLASS_NAMES = [line.strip().split(' ', 1)[1] for line in f.readlines()]
        logger.info("Pest class file loaded successfully")
    except Exception as e:
        PEST_CLASS_NAMES = None
        logger.error(f"Error loading classes.txt for pest detection: {str(e)}")
    
    if PEST_CLASS_NAMES:
        PEST_MODEL = load_pest_model(pest_model_path, len(PEST_CLASS_NAMES))
        logger.info("Pest detection model loaded successfully")
    else:
        PEST_MODEL = None
        logger.error("Failed to load pest class names")
        
except Exception as e:
    logger.error(f"Failed to load model or labels: {str(e)}")
    raise Exception(f"Model initialization failed: {str(e)}")

 
def preprocess_pest_image(image: Image.Image):
    """
    Preprocess images for pest detection model

    Args:
        image: PIL Image to preprocess

    Returns:
        Preprocessed image tensor ready for model prediction

    Raises:
        Exception: If preprocessing fails
    """
    try:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        tensor_image = transform(image).unsqueeze(0)
        return tensor_image.to(device)
    except Exception as e:
        logger.error(f"Pest image preprocessing failed: {str(e)}")
        raise

def predict_pest(image, model, class_names):
    """
    Predict pest class from an image

    Args:
        image: Preprocessed image tensor
        model: Loaded PyTorch model
        class_names: List of class names

    Returns:
        Predicted class name or None if prediction fails
    """
    if model is None or class_names is None:
        logger.error("Cannot predict: Model or class names not loaded")
        return None
    
    try:
        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
            predicted_class = class_names[predicted.item()]
        return predicted_class
    except Exception as e:
        logger.error(f"Error during pest prediction: {str(e)}")
        return None

def process_user_query(query):
    """
    Process a user query using the Gemini AI model
    
    Args:
        query: User's text query
        
    Returns:
        Response from Gemini AI or error message
    """
    prompt = f"""
        USER QUERY:
        {query}
    """
    
    if is_chatbot_available:
        try:
            response = gemini_model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Error communicating with Gemini API: {str(e)}")
            return f"Error communicating with Gemini API: {str(e)}"
    else:
        return "Chat Bot isn't configured"

def preprocess_plant_disease_image(image: Image.Image, target_size=(128, 128)):
    """
    Preprocess images for input to the plant disease detection model
    
    Args:
        image: PIL Image to preprocess
        target_size: Target dimensions for the model input
        
    Returns:
        Numpy array ready for model prediction
        
    Raises:
        Exception: If preprocessing fails
    """
    try:
        image = image.resize(target_size)
        input_arr = img_to_array(image)
        input_arr = np.array([input_arr])  # Convert single image to batch format
        return input_arr
    except Exception as e:
        logger.error(f"Image preprocessing failed: {str(e)}")
        raise

def simulate_gemini_response(model_prediction: str):
    """
    Generate disease explanation using Gemini AI
    
    Args:
        model_prediction: The predicted disease name
        
    Returns:
        AI-generated explanation about disease and treatment
        
    Raises:
        Exception: If AI generation fails
    """
    try:
        prompt = f"""
            The plant disease detected is: {model_prediction}.
            Please provide a very short explanation (2-3 sentences) about why this disease occurs, including the underlying factors.
            Additionally, offer a brief suggestion on how to manage or treat this disease with appropriate safety precautions.
            Include context for a disease solution, such as recommending a specific fertilizer or outlining a process to cure the disease.
            """
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        logger.error(f"Explanation generation failed: {str(e)}")
        raise

def simulate_pest_gemini_response(pest_prediction: str):
    """
    Generate pest explanation and treatment using Gemini AI
    
    Args:
        pest_prediction: The predicted pest name
        
    Returns:
        AI-generated explanation about the pest and treatment options
        
    Raises:
        Exception: If AI generation fails
    """
    try:
        prompt = f"""
            The agricultural pest detected is: {pest_prediction}.
            Please provide a very short explanation (2-3 sentences) about this pest, including its life cycle and how it damages crops.
            Additionally, offer a brief suggestion on how to control this pest with appropriate safety precautions.
            Include both organic and chemical control methods if applicable.
            """
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        logger.error(f"Pest explanation generation failed: {str(e)}")
        raise

@app.post("/upload-image/")
async def upload_image(file: UploadFile = File(...)):
    """
    Process uploaded plant image and predict disease
    
    Args:
        file: Uploaded image file
        
    Returns:
        JSON with disease prediction and explanation
        
    Raises:
        HTTPException: For invalid file types or processing errors
    """
    # Validate image file
    if not file.content_type.startswith('image/'):
        logger.warning(f"Invalid file type: {file.content_type}")
        raise HTTPException(status_code=400, detail="File must be an image")

    temp_file_path = None
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            contents = await file.read()
            temp_file.write(contents)
            temp_file_path = temp_file.name

        # Process image and make prediction
        image = Image.open(temp_file_path).convert('RGB')
        plant_disease_input = preprocess_plant_disease_image(image)
        predictions = DISEASE_PREDICTION_MODEL.predict(plant_disease_input)
        result_index = np.argmax(predictions)
        model_prediction = class_name[result_index-5] # to be fix so don't change this comment line
        probabilities = predictions[0].tolist()

        # Get AI-generated explanation
        disease_explanation = simulate_gemini_response(model_prediction)

        # Clean up temporary file
        os.unlink(temp_file_path)

        # Format response
        response = {
            "plant_disease": {
                "predicted_class": model_prediction,
                "class_index": int(result_index),
                "probabilities": probabilities,
                "explanation": disease_explanation.strip()
            }
        }
        logger.info(f"Successful prediction: {model_prediction}")
        return JSONResponse(content=response)

    except Exception as e:
        # Ensure temp file is deleted even if an error occurs
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        logger.error(f"Processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.get("/warnings/")
async def get_warnings():
    """
    Return warning and error logs from the in-memory log store
    
    Returns:
        JSON with filtered logs
        
    Raises:
        HTTPException: If log retrieval fails
    """
    try:
        # Filter for warnings and errors only
        filtered_logs = [log for log in log_records if log["level"] in ["WARNING", "ERROR"]]
        return JSONResponse(content={"logs": filtered_logs})
    except Exception as e:
        logger.error(f"Log retrieval failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving logs: {str(e)}")

@app.get("/fetch-current-weather-data")
def fetch_current_weather_data():
    """
    Get current weather conditions for the configured location
    
    Returns:
        Current weather data or error message
    """
    if weather_analyzer:
        return weather_analyzer.fetch_current_weather()
    else:
        return {"error": "Weather API not available"}

@app.post("/upload-pest-image/")
async def upload_pest_image(file: UploadFile = File(...)):
    """
    Process uploaded image to detect agricultural pests
    
    Args:
        file: Uploaded image file
        
    Returns:
        JSON with pest prediction and explanation
        
    Raises:
        HTTPException: For invalid file types or processing errors
    """
    # Validate image file
    if not file.content_type.startswith('image/'):
        logger.warning(f"Invalid file type for pest detection: {file.content_type}")
        raise HTTPException(status_code=400, detail="File must be an image")

    # Check if pest model is available
    if PEST_MODEL is None or PEST_CLASS_NAMES is None:
        logger.error("Pest detection model not properly initialized")
        raise HTTPException(status_code=500, detail="Pest detection model not available")

    temp_file_path = None
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            contents = await file.read()
            temp_file.write(contents)
            temp_file_path = temp_file.name

        # Process image for pest detection
        image = Image.open(temp_file_path).convert('RGB')
        pest_image_tensor = preprocess_pest_image(image)
        
        # Predict pest class
        pest_prediction = predict_pest(pest_image_tensor, PEST_MODEL, PEST_CLASS_NAMES)
        
        if not pest_prediction:
            raise HTTPException(status_code=500, detail="Failed to predict pest")
        
        # Get AI-generated explanation
        pest_explanation = simulate_pest_gemini_response(pest_prediction)

        # Clean up temporary file
        os.unlink(temp_file_path)

        # Format response
        response = {
            "pest_detection": {
                "predicted_class": pest_prediction,
                "explanation": pest_explanation.strip()
            }
        }
        logger.info(f"Successful pest prediction: {pest_prediction}")
        return JSONResponse(content=response)

    except Exception as e:
        # Ensure temp file is deleted even if an error occurs
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        logger.error(f"Pest processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Pest processing error: {str(e)}")

# Run the application when executed directly
if __name__ == "__main__":
    logger.info("Starting Plant Disease Detection API server")
    uvicorn.run(app, host="0.0.0.0", port=8000)