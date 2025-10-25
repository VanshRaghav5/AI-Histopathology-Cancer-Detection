# Histopathology Cancer Detection FastAPI Server

A FastAPI server for detecting cancer in histopathology images using a trained CNN model. This server provides REST API endpoints for image classification and can be easily integrated with JavaScript frontend applications.

## Features

- **Binary Classification**: Detects benign vs malignant histopathology images
- **REST API**: Easy integration with web applications
- **Batch Processing**: Support for multiple image predictions
- **CORS Enabled**: Ready for JavaScript frontend integration
- **Model Loading**: Automatic model loading on startup
- **Error Handling**: Comprehensive error handling and logging

## Model Information

- **Architecture**: MobileNetV2 (configurable)
- **Input Size**: 224x224 pixels
- **Classes**: 
  - 0: Benign
  - 1: Malignant
- **Output**: Prediction probabilities and confidence scores

## Installation

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd histopathic_fastapi_server
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare your trained model**:
   - Place your trained model file as `models/best_model.pth`
   - The model should be saved using the checkpoint format from the training script

## Usage

### Starting the Server

```bash
python app.py
```

The server will start on `http://localhost:8000`

### API Endpoints

#### 1. Health Check
```http
GET /health
```

**Response**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda:0"
}
```

#### 2. Single Image Prediction
```http
POST /predict
Content-Type: multipart/form-data
```

**Request**: Upload an image file

**Response**:
```json
{
  "prediction": "malignant",
  "confidence": 0.9234,
  "probabilities": {
    "benign": 0.0766,
    "malignant": 0.9234
  },
  "class_id": 1,
  "filename": "image.jpg",
  "file_size": 245760,
  "content_type": "image/jpeg"
}
```

#### 3. Batch Prediction
```http
POST /predict-batch
Content-Type: multipart/form-data
```

**Request**: Upload multiple image files (max 10)

**Response**:
```json
{
  "predictions": [
    {
      "prediction": "benign",
      "confidence": 0.8567,
      "probabilities": {
        "benign": 0.8567,
        "malignant": 0.1433
      },
      "class_id": 0,
      "filename": "image1.jpg",
      "file_size": 245760,
      "content_type": "image/jpeg"
    }
  ]
}
```

#### 4. Model Information
```http
GET /model-info
```

**Response**:
```json
{
  "model_architecture": "MobileNetV2",
  "num_classes": 2,
  "classes": ["benign", "malignant"],
  "input_size": [224, 224],
  "device": "cuda:0",
  "model_loaded": true
}
```

## JavaScript Integration

### Using Fetch API

```javascript
// Single image prediction
async function predictImage(imageFile) {
  const formData = new FormData();
  formData.append('file', imageFile);
  
  try {
    const response = await fetch('http://localhost:8000/predict', {
      method: 'POST',
      body: formData
    });
    
    const result = await response.json();
    console.log('Prediction:', result);
    return result;
  } catch (error) {
    console.error('Error:', error);
  }
}

// Batch prediction
async function predictBatch(imageFiles) {
  const formData = new FormData();
  imageFiles.forEach(file => {
    formData.append('files', file);
  });
  
  try {
    const response = await fetch('http://localhost:8000/predict-batch', {
      method: 'POST',
      body: formData
    });
    
    const result = await response.json();
    console.log('Batch predictions:', result);
    return result;
  } catch (error) {
    console.error('Error:', error);
  }
}
```

### Using Axios

```javascript
import axios from 'axios';

const api = axios.create({
  baseURL: 'http://localhost:8000',
  headers: {
    'Content-Type': 'multipart/form-data'
  }
});

// Single prediction
async function predictImage(imageFile) {
  const formData = new FormData();
  formData.append('file', imageFile);
  
  try {
    const response = await api.post('/predict', formData);
    return response.data;
  } catch (error) {
    console.error('Error:', error);
  }
}
```

## Configuration

### Environment Variables

- `MODEL_PATH`: Path to the model file (default: `models/best_model.pth`)
- `PORT`: Server port (default: 8000)
- `HOST`: Server host (default: 0.0.0.0)

### CORS Configuration

The server is configured to allow all origins for development. For production, update the CORS settings in `app.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Your frontend domain
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

## File Structure

```
histopathic_fastapi_server/
├── app.py                 # FastAPI server
├── train.py              # Training script
├── requirements.txt      # Dependencies
├── README.md            # This file
├── models/              # Model files
│   └── best_model.pth   # Trained model
└── src/
    └── model/
        ├── __init__.py
        ├── model.py     # Model architecture
        ├── dataset.py   # Dataset utilities
        └── utils.py     # Utility functions
```

## Training the Model

To train a new model, use the provided training script:

```bash
python train.py --data_dir data --epochs 15 --batch_size 32 --lr 0.001
```

Training arguments:
- `--data_dir`: Path to data directory
- `--arch`: Model architecture (default: mobilenet_v2)
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size
- `--lr`: Learning rate
- `--pretrained`: Use pretrained weights
- `--save_dir`: Directory to save models

## Error Handling

The API includes comprehensive error handling:

- **400 Bad Request**: Invalid image format or file type
- **503 Service Unavailable**: Model not loaded
- **500 Internal Server Error**: Server-side errors

## Performance Notes

- The model automatically uses GPU if available
- Images are resized to 224x224 pixels
- Batch processing is limited to 10 images per request
- Model is loaded once at startup for optimal performance

## Development

### Running in Development Mode

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### API Documentation

Once the server is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## License

This project is licensed under the MIT License.
