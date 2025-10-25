#!/usr/bin/env python3
"""
Test script for the Histopathology Cancer Detection API
"""

import requests
import os
from PIL import Image
import io
import numpy as np

def create_test_image(size=(224, 224), color='RGB'):
    """Create a test image for API testing"""
    # Create a random test image
    if color == 'RGB':
        image_array = np.random.randint(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    else:
        image_array = np.random.randint(0, 255, (size[1], size[0]), dtype=np.uint8)
    
    image = Image.fromarray(image_array)
    
    # Save to bytes
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    
    return img_bytes

def test_health_endpoint():
    """Test the health check endpoint"""
    print("Testing health endpoint...")
    try:
        response = requests.get("http://localhost:8000/health")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_model_info():
    """Test the model info endpoint"""
    print("\nTesting model info endpoint...")
    try:
        response = requests.get("http://localhost:8000/model-info")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_single_prediction():
    """Test single image prediction"""
    print("\nTesting single image prediction...")
    try:
        # Create test image
        test_image = create_test_image()
        
        # Prepare request
        files = {'file': ('test_image.jpg', test_image, 'image/jpeg')}
        
        response = requests.post("http://localhost:8000/predict", files=files)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_batch_prediction():
    """Test batch image prediction"""
    print("\nTesting batch image prediction...")
    try:
        # Create multiple test images
        test_images = []
        for i in range(3):
            img = create_test_image()
            test_images.append(('files', (f'test_image_{i}.jpg', img, 'image/jpeg')))
        
        response = requests.post("http://localhost:8000/predict-batch", files=test_images)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_invalid_file():
    """Test with invalid file type"""
    print("\nTesting invalid file type...")
    try:
        # Create a text file instead of image
        files = {'file': ('test.txt', io.BytesIO(b'This is not an image'), 'text/plain')}
        
        response = requests.post("http://localhost:8000/predict", files=files)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 400  # Should return 400 for invalid file
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    """Run all tests"""
    print("Starting API tests...")
    print("=" * 50)
    
    tests = [
        test_health_endpoint,
        test_model_info,
        test_single_prediction,
        test_batch_prediction,
        test_invalid_file
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print("-" * 30)
    
    print(f"\nTest Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("All tests passed! ✅")
    else:
        print("Some tests failed! ❌")

if __name__ == "__main__":
    main()
