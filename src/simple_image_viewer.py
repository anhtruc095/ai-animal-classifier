#!/usr/bin/env python3
"""
Simple Image Viewer with Predictions
Shows images and their predictions using the high-accuracy model
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

def load_model():
    """Load the high-accuracy model"""
    model_path = "checkpoints/resnet18_animals10.pth"
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        return None
    
    try:
        # Create ResNet18 model with EXACT same architecture as training
        model = models.resnet18(pretrained=False)
        num_features = model.fc.in_features
        
        # Match the exact architecture from training - Sequential with Dropout and two Linear layers
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 10)
        )
        
        # Load the saved weights
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint)
        model.eval()
        print(f"✅ Model loaded successfully!")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Trying to display image without prediction...")
        return None

def predict_image(model, image_path):
    """Make prediction on an image"""
    if model is None:
        return None, None, []
    
    # Class names (same order as training)
    class_names = [
        'butterfly', 'cat', 'chicken', 'cow', 'dog', 
        'elephant', 'horse', 'sheep', 'spider', 'squirrel'
    ]
    
    # Image preprocessing (same as training)
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
            
            # Get top 3 predictions
            top3_prob, top3_idx = torch.topk(probabilities, 3, dim=1)
            
            predicted_class = class_names[top3_idx[0][0].item()]
            confidence_score = top3_prob[0][0].item() * 100
            
            # Get top 3 results
            top3_results = []
            for i in range(3):
                class_name = class_names[top3_idx[0][i].item()]
                prob = top3_prob[0][i].item() * 100
                top3_results.append((class_name, prob))
            
            return predicted_class, confidence_score, top3_results
    
    except Exception as e:
        print(f"❌ Error predicting image: {e}")
        return None, None, []

def show_image_with_prediction(image_path, model=None):
    """Display image with prediction"""
    try:
        # Create figure with larger size
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Load and display image
        if image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = mpimg.imread(image_path)
        else:
            pil_img = Image.open(image_path)
            img = np.array(pil_img)
        
        ax.imshow(img)
        ax.axis('off')
        
        # Create title with image name
        image_name = os.path.basename(image_path)
        title = f"🖼️  {image_name}"
        
        # Add prediction if model is available
        if model:
            predicted_class, confidence, top3 = predict_image(model, image_path)
            if predicted_class:
                title += f"\n\n🎯 Prediction: {predicted_class.upper()}"
                title += f"\n📊 Confidence: {confidence:.1f}%"
                
                if top3:
                    title += f"\n\n📋 Top 3 Predictions:"
                    for i, (class_name, prob) in enumerate(top3, 1):
                        title += f"\n   {i}. {class_name}: {prob:.1f}%"
        
        ax.set_title(title, fontsize=12, pad=20, loc='left')
        
        # Adjust layout and show
        plt.tight_layout()
        plt.show()
        
        # Also print prediction to terminal
        if model and predicted_class:
            print(f"\n🎯 Prediction Result:")
            print(f"   Image: {image_name}")
            print(f"   Predicted: {predicted_class.upper()}")
            print(f"   Confidence: {confidence:.1f}%")
            print(f"\n📋 Top 3 Predictions:")
            for i, (class_name, prob) in enumerate(top3, 1):
                print(f"   {i}. {class_name}: {prob:.1f}%")
        
    except Exception as e:
        print(f"❌ Error displaying image: {e}")

def list_sample_images():
    """List some sample images from each class"""
    data_dir = Path("data/test")
    if not data_dir.exists():
        print("❌ Test data directory not found")
        return
    
    print("🖼️  Sample images available for viewing:")
    print("=" * 60)
    
    for class_dir in sorted(data_dir.iterdir()):
        if class_dir.is_dir():
            images = list(class_dir.glob("*.jpeg"))[:3]  # Show first 3 images
            if images:
                print(f"\n🐾 {class_dir.name.upper()}:")
                for i, img_path in enumerate(images, 1):
                    print(f"   {i}. {img_path}")

def main():
    parser = argparse.ArgumentParser(description="View images with AI predictions")
    parser.add_argument("--image", type=str, help="Path to image file")
    parser.add_argument("--list", action="store_true", help="List available sample images")
    parser.add_argument("--no-predict", action="store_true", help="Show image without predictions")
    
    args = parser.parse_args()
    
    # Change to project directory
    os.chdir(Path(__file__).parent.parent)
    
    if args.list:
        list_sample_images()
        return
    
    if not args.image:
        print("🖼️  AI Animal Image Viewer")
        print("=" * 40)
        print("This tool shows images with AI predictions from your trained model!")
        print("\nUsage:")
        print("  python src/simple_image_viewer.py --image <path_to_image>")
        print("  python src/simple_image_viewer.py --list")
        print("\nExamples:")
        print("  python src/simple_image_viewer.py --image data/test/cat/298.jpeg")
        print("  python src/simple_image_viewer.py --image data/test/dog/OIP-P3zTu-bZLnRAvvups-MEFQHaE6.jpeg")
        print("  python src/simple_image_viewer.py --image data/test/butterfly/OIP-BtCgHCaDIDG52nLmqVF0nwHaFj.jpeg")
        return
    
    # Check if image exists
    if not os.path.exists(args.image):
        print(f"❌ Image not found: {args.image}")
        return
    
    # Load model unless disabled
    model = None
    if not args.no_predict:
        print("📥 Loading AI model for predictions...")
        model = load_model()
        if model:
            print("✅ Ready to make predictions!")
        else:
            print("⚠️  Will show image without predictions")
    
    # Show image with prediction
    print(f"🖼️  Displaying: {args.image}")
    show_image_with_prediction(args.image, model)

if __name__ == "__main__":
    main()