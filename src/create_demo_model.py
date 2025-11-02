import torch
from torchvision import models
import os

# Create a demo model for testing prediction
print("Creating demo model...")

# Same architecture as training
model = models.resnet18(pretrained=True)
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, 10)  # 10 classes

# Get the correct path
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
save_dir = os.path.join(base_dir, 'checkpoints')
os.makedirs(save_dir, exist_ok=True)
model_path = os.path.join(save_dir, 'resnet18_animals10.pth')

# Save the model (this will be a pre-trained model, not fully trained on our data)
torch.save(model.state_dict(), model_path)
print(f"Demo model saved to: {model_path}")
print("Note: This is a demo model. For best results, run the full training with: python src/train.py")