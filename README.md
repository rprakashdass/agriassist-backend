# AgriAssist - Agricultural Disease & Pest Detection API

AgriAssist is a FastAPI-based service designed to help farmers and gardeners identify plant diseases and agricultural pests through image recognition and provide relevant treatment recommendations. It also integrates weather data analysis to assist in agricultural decision-making.

## Features

- Plant disease detection from images with 38 disease classes
- Agricultural pest identification with 102 pest classes
- AI-powered explanations and treatment recommendations
- Real-time weather data for agricultural planning and pest risk assessment
- Warning and error tracking system

## API Endpoints

### 1. Plant Disease Detection

**Endpoint:** `POST /upload-image/`

Upload a plant image to detect diseases and get treatment recommendations.

#### Request

- Format: `multipart/form-data`
- Field: `file` (image file - JPG, PNG, etc.)

#### Response

```json
{
  "plant_disease": {
    "predicted_class": "Tomato_Late_blight",
    "class_index": 32,
    "probabilities": [0.001, 0.002, ..., 0.985, ...],
    "explanation": "Late blight is caused by the oomycete pathogen Phytophthora infestans that thrives in cool, wet conditions. It spreads rapidly through water splashes and wind. To manage this disease, apply copper-based fungicides early in the season, ensure proper plant spacing for air circulation, and water at the base of plants in the morning to allow foliage to dry quickly."
  }
}
```

### 2. Weather Information

**Endpoint:** `GET /fetch-current-weather-data`

Fetches current weather data for the configured location (default: Coimbatore).

#### Response

```json
{
  "location": {
    "name": "Coimbatore",
    "region": "Tamil Nadu",
    "country": "India"
  },
  "current": {
    "temp_c": 29.0,
    "temp_f": 84.2,
    "condition": {
      "text": "Partly cloudy",
      "icon": "//cdn.weatherapi.com/weather/64x64/day/116.png"
    },
    "humidity": 65,
    "cloud": 25,
    "feelslike_c": 30.7,
    "feelslike_f": 87.3,
    "precip_mm": 0.0
  }
}
```

### 3. Pest Detection

**Endpoint:** `POST /upload-pest-image/`

Upload an image to detect agricultural pests and get treatment recommendations.

#### Request

- Format: `multipart/form-data`
- Field: `file` (image file - JPG, PNG, etc.)

#### Response

```json
{
  "pest_detection": {
    "predicted_class": "Rice Stem Borer",
    "explanation": "Rice stem borers are moths whose larvae bore into rice stems during their development, causing yellowing and drying of the central leaf, known as 'deadheart' in young plants and 'whitehead' in older plants. To control them, practice proper field sanitation by removing rice stubble after harvest, use resistant rice varieties, and apply biological controls like parasitoids. Chemical control with appropriate insecticides may be necessary for severe infestations, but always follow safety guidelines and consider integrated pest management approaches."
  }
}
```

### 4. System Warnings

**Endpoint:** `GET /warnings/`

Retrieve system warnings and errors for monitoring.

#### Response

```json
{
  "logs": [
    {
      "timestamp": "2025-05-31T14:32:45.123456",
      "level": "WARNING",
      "message": "Invalid file type uploaded: text/plain"
    },
    {
      "timestamp": "2025-05-31T15:01:22.654321",
      "level": "ERROR",
      "message": "Processing error: Failed to load image"
    }
  ]
}
```

## Datasets

### Plant Disease Dataset

The plant disease detection model is trained on a comprehensive dataset containing approximately 87,000 RGB images of healthy and diseased crop leaves categorized into 38 different classes.

- **Source**: [New Plant Diseases Dataset on Kaggle](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
- **Classes**: 38 different classes of plant diseases and healthy plants
- **Image Size**: 128 x 128 pixels
- **Training/Validation Split**: 80% training, 20% validation
- **Preprocessing**: RGB image normalization and resizing

### Pest Detection Dataset (IP102)

The pest identification model is trained on the IP102 dataset containing over 75,000 images of different agricultural pests.

- **Source**: [IP102 Dataset on Kaggle](https://www.kaggle.com/datasets/rtlmhjbn/ip02-dataset)
- **Classes**: 102 different pest species organized into 8 super-classes:
  - **Field Crops (FC)**: Rice, Corn, Wheat, Beet, Alfalfa
  - **Economic Crops (EC)**: Vitis, Citrus, Mango
- **Images**: 75,222 images with an average of 737 samples per class
- **Split**: 60% training, 10% validation, 30% testing
- **Image Size**: 224 x 224 pixels (resized)
- **Reference**: Wu, X., Zhan, C., Lai, Y.K., Cheng, M.M., & Yang, J. (2019). CVPR 2019

## Models

### Plant Disease Detection Model

A convolutional neural network trained to identify plant diseases from leaf images.

- **Architecture**: Custom CNN with multiple convolutional and pooling layers
- **Input**: 128 x 128 RGB images
- **Output**: 38 classes (various plant diseases and healthy plants)
- **Training Details**: See `pyNoteBooks/plantDiseaseDetectionTraining.ipynb`

### Pest Detection Model

A deep learning model trained to identify agricultural pests.

- **Architecture**: ResNet50
- **Input**: 224 x 224 RGB images
- **Output**: 102 pest classes
- **Training Details**: See `pyNoteBooks/pest_detection_train.ipynb`

## Installation and Setup

### Prerequisites

- Python 3.9+
- TensorFlow 2.x
- PyTorch
- FastAPI
- Other dependencies in `requirements.txt`

### Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/AgriAssist.git
cd AgriAssist
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set up environment variables in a `.env` file:

```
WEATHER_API_KEY=your_weather_api_key
GEMINI_API_KEY=your_gemini_api_key
```

4. Run the application:

```bash
python main.py 
# or
uvicorn main:app --host 0.0.0.0 --port 8000
```

## API Documentation

When running the application, access the auto-generated API documentation at:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Model Training

### Plant Disease Detection Model Training

The plant disease detection model was trained using TensorFlow on the New Plant Diseases Dataset. The model architecture consists of multiple convolutional and pooling layers followed by fully connected layers.

Training details:

- Input image size: 128x128 pixels
- Optimizer: Adam with learning rate 0.0001
- Loss function: Categorical Cross-Entropy
- Batch size: 32
- Epochs: 10+
- Architecture: Multi-layer CNN with dropout to prevent overfitting

See the complete training process in the notebook: `pyNoteBooks/plantDiseaseDetectionTraining.ipynb`

### Pest Detection Model Training

The agricultural pest detection model uses a fine-tuned ResNet50 architecture trained on the IP102 dataset using PyTorch.

Training details:

- Input image size: 224x224 pixels
- Model: ResNet50 with modified fully connected layer
- Optimizer: SGD with momentum (0.9) and weight decay (1e-4)
- Loss function: Cross-Entropy Loss
- Batch size: 32
- Data augmentation: Random horizontal flip, random rotation

See the complete training process in the notebook: `pyNoteBooks/pest_detection_train.ipynb`

## System Architecture

AgriAssist integrates multiple components:

1. **FastAPI Backend**: Handles HTTP requests and serves predictions
2. **TensorFlow Model**: Detects plant diseases using a custom CNN architecture
3. **PyTorch Model**: Identifies agricultural pests using a ResNet50 architecture
4. **Gemini AI**: Provides natural language explanations and treatment recommendations
5. **Weather API**: Provides real-time weather data for agricultural decisions
6. **Logging System**: Tracks warnings and errors for monitoring and diagnostics

The system is designed to be modular, allowing for easy updates of individual components without affecting others.

## Project Structure

```
AgriAssist/
├── main.py                  # Main API application
├── WeatherAnalysis.py       # Weather data analysis module
├── requirements.txt         # Project dependencies
├── README.md                # Project documentation
├── Labels/                  # Model labels
│   ├── classes.txt          # Pest class names
│   └── labels.csv           # Plant disease class names
├── models/                  # Trained models
│   ├── resnet50_ip102.pth   # PyTorch pest detection model
│   └── trained_plant_disease_model.keras  # TensorFlow plant disease model
├── pyNoteBooks/             # Training & testing notebooks
│   ├── pest_detection_test.ipynb
│   ├── pest_detection_train.ipynb
│   ├── plantDiseaseDetectionTesting.ipynb
│   └── plantDiseaseDetectionTraining.ipynb
└── test_images/             # Test images for verification
```

## Data Flow

1. User uploads an image through one of the API endpoints
2. Image is preprocessed based on the target model requirements
3. Appropriate model (plant disease or pest) makes a prediction
4. Prediction results are sent to Gemini AI to generate human-friendly explanations
5. Combined results (prediction + explanation) are returned to the user
6. All events are logged for monitoring and improvement
