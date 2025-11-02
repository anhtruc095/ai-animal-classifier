#!/usr/bin/env python3
"""
Image Viewer for Animals Classifier Dataset
Shows images with predictions from the high-accuracy model
"""

import argparse
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
import numpy as np

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent))

def load_model():
    """Load the trained model with correct architecture"""
    model_path = "checkpoints/resnet18_animals10.pth"
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        return None
    
    try:
        # Create model with the same architecture as training
        model = models.resnet18(pretrained=False)
        num_features = model.fc.in_features
        
        # Replace the classifier with Sequential layers (matching training script)
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 10)  # 10 classes
        )
        
        # Load the saved weights
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint)
        model.eval()
        print(f"✅ Model loaded from {model_path}")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None



def predict_image(model, image_path):
    """Make prediction on an image"""
    if model is None:
        return None, None
    
    # Class names mapping
    class_names = [
        'butterfly', 'cat', 'chicken', 'cow', 'dog', 
        'elephant', 'horse', 'sheep', 'spider', 'squirrel'
    ]
    
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        
        # Make prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            predicted_class = class_names[predicted.item()]
            confidence_score = confidence.item() * 100
            
            return predicted_class, confidence_score
    
    except Exception as e:
        print(f"❌ Error predicting image: {e}")
        return None, None

def show_image_with_prediction(image_path, model=None):
    """Display image with prediction"""
    try:
        # Load image
        if image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = mpimg.imread(image_path)
        else:
            # Use PIL for other formats
            pil_img = Image.open(image_path)
            img = np.array(pil_img)
        
        # Create figure
        plt.figure(figsize=(10, 8))
        plt.imshow(img)
        plt.axis('off')
        
        # Add prediction if model is available
        title = f"Image: {os.path.basename(image_path)}"
        if model:
            predicted_class, confidence = predict_image(model, image_path)
            if predicted_class:
                title += f"\n🎯 Prediction: {predicted_class} ({confidence:.1f}% confidence)"
        
        plt.title(title, fontsize=14, pad=20)
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"❌ Error displaying image: {e}")

def list_sample_images():
    """List some sample images from each class"""
    data_dir = Path("data/test")
    if not data_dir.exists():
        print("❌ Test data directory not found")
        return
    
    print("📁 Sample images available:")
    print("=" * 50)
    
    for class_dir in sorted(data_dir.iterdir()):
        if class_dir.is_dir():
            images = list(class_dir.glob("*.jpeg"))[:3]  # Show first 3 images
            print(f"\n🐾 {class_dir.name.upper()}:")
            for i, img_path in enumerate(images, 1):
                print(f"  {i}. {img_path}")

def main():
    parser = argparse.ArgumentParser(description="Show images from the animals dataset")
    parser.add_argument("--image", type=str, help="Path to image file")
    parser.add_argument("--list", action="store_true", help="List sample images")
    parser.add_argument("--no-predict", action="store_true", help="Don't load model for predictions")
    
    args = parser.parse_args()
    
    # Change to project directory
    os.chdir(Path(__file__).parent.parent)
    
    if args.list:
        list_sample_images()
        return
    
    if not args.image:
        print("🖼️  Animal Image Viewer")
        print("=" * 30)
        print("Usage:")
        print("  python src/show_image.py --image <path_to_image>")
        print("  python src/show_image.py --list  # Show available images")
        print("\nExamples:")
        print("  python src/show_image.py --image data/test/cat/OIF-e2bexWrojgtQnAPPcUfOWQ.jpeg")
        print("  python src/show_image.py --image data/test/dog/123.jpg")
        return
    
    # Load model unless disabled
    model = None
    if not args.no_predict:
        print("📥 Loading model for predictions...")
        model = load_model()
    
    # Show image
    if os.path.exists(args.image):
        print(f"🖼️  Displaying: {args.image}")
        show_image_with_prediction(args.image, model)
    else:
        print(f"❌ Image not found: {args.image}")

if __name__ == "__main__":
    main()