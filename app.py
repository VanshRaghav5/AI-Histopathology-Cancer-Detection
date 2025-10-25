from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import io
import numpy as np
from typing import Dict, Any
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Histopathology Cancer Detection API",
    description="API for detecting cancer in histopathology images using CNN",
    version="1.0.0"
)

# Add CORS middleware for JavaScript frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and device
model = None
device = None
transform = None

class HistopathologyModel(nn.Module):
    """Wrapper for the histopathology detection model"""
    def __init__(self, num_classes=2):
        super(HistopathologyModel, self).__init__()
        # This will be replaced with the actual model architecture
        # For now, we'll load the saved model directly
        pass

def load_model(model_path: str = "models/best_model.pth"):
    """Load the trained model"""
    global model, device, transform
    
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        # Load model architecture (you'll need to import your actual model)
        # For now, we'll assume the model is saved with state_dict
        from src.model.model import get_model
        
        model = get_model(name="mobilenet_v2", num_classes=2, pretrained=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        # Define image preprocessing pipeline
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        logger.info("Model loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False

def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    """Preprocess uploaded image for model inference"""
    try:
        # Open image from bytes
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply transforms
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        return image_tensor.to(device)
        
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid image format")

def predict_cancer(image_tensor: torch.Tensor) -> Dict[str, Any]:
    """Make prediction on preprocessed image"""
    try:
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            
            # Get prediction probabilities
            benign_prob = probabilities[0][0].item()
            malignant_prob = probabilities[0][1].item()
            
            # Get predicted class
            predicted_class = torch.argmax(probabilities, dim=1).item()
            class_name = "benign" if predicted_class == 0 else "malignant"
            
            # Calculate confidence
            confidence = max(benign_prob, malignant_prob)
            
            return {
                "prediction": class_name,
                "confidence": round(confidence, 4),
                "probabilities": {
                    "benign": round(benign_prob, 4),
                    "malignant": round(malignant_prob, 4)
                },
                "class_id": int(predicted_class)
            }
            
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail="Prediction failed")

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    model_path = os.getenv("MODEL_PATH", "models/best_model.pth")
    if not load_model(model_path):
        logger.warning("Model loading failed. Please ensure model file exists.")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Histopathology Cancer Detection API",
        "status": "running",
        "model_loaded": model is not None
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device) if device else None
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict cancer in histopathology image
    
    Args:
        file: Image file (JPEG, PNG, etc.)
    
    Returns:
        JSON with prediction results
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image bytes
        image_bytes = await file.read()
        
        # Preprocess image
        image_tensor = preprocess_image(image_bytes)
        
        # Make prediction
        result = predict_cancer(image_tensor)
        
        # Add metadata
        result.update({
            "filename": file.filename,
            "file_size": len(image_bytes),
            "content_type": file.content_type
        })
        
        return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/predict-batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    """
    Predict cancer in multiple histopathology images
    
    Args:
        files: List of image files
    
    Returns:
        JSON with prediction results for each image
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(files) > 10:  # Limit batch size
        raise HTTPException(status_code=400, detail="Maximum 10 files allowed per batch")
    
    results = []
    
    for file in files:
        try:
            # Validate file type
            if not file.content_type.startswith('image/'):
                results.append({
                    "filename": file.filename,
                    "error": "Invalid file type"
                })
                continue
            
            # Read image bytes
            image_bytes = await file.read()
            
            # Preprocess image
            image_tensor = preprocess_image(image_bytes)
            
            # Make prediction
            result = predict_cancer(image_tensor)
            result.update({
                "filename": file.filename,
                "file_size": len(image_bytes),
                "content_type": file.content_type
            })
            
            results.append(result)
            
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e)
            })
    
    return JSONResponse(content={"predictions": results})

@app.get("/model-info")
async def model_info():
    """Get information about the loaded model"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_architecture": "MobileNetV2",
        "num_classes": 2,
        "classes": ["benign", "malignant"],
        "input_size": [224, 224],
        "device": str(device),
        "model_loaded": True
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
