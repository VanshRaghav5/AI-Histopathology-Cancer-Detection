#!/usr/bin/env python3
"""
Simple script to run the FastAPI server
"""

import uvicorn
import os
import sys

def main():
    """Run the FastAPI server"""
    # Check if model directory exists
    if not os.path.exists("models"):
        print("Creating models directory...")
        os.makedirs("models")
        print("Please place your trained model as 'models/best_model.pth'")
    
    # Check if model file exists
    model_path = "models/best_model.pth"
    if not os.path.exists(model_path):
        print(f"Warning: Model file not found at {model_path}")
        print("The server will start but predictions will fail until a model is provided.")
    
    # Run the server
    print("Starting Histopathology Cancer Detection API server...")
    print("Server will be available at: http://localhost:8000")
    print("API documentation: http://localhost:8000/docs")
    print("Press Ctrl+C to stop the server")
    
    try:
        uvicorn.run(
            "app:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
