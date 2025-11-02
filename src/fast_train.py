import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import os

print("🚀 Starting FAST training for immediate results...")

# Get paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
train_dir = os.path.join(base_dir, 'data', 'train')
test_dir = os.path.join(base_dir, 'data', 'test')

# Quick transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Create datasets
print("📁 Loading dataset...")
train_dataset = datasets.ImageFolder(train_dir, transform=transform)
test_dataset = datasets.ImageFolder(test_dir, transform=transform)

# Smaller batch size for speed
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0)

print(f"✅ Dataset loaded: {len(train_dataset)} training images, {len(test_dataset)} test images")
print(f"📋 Classes: {train_dataset.classes}")

# Create model
print("🧠 Creating model...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"💻 Training on: {device}")

model = models.resnet18(pretrained=True)
# Freeze early layers but keep some trainable for better results
for i, (name, param) in enumerate(model.named_parameters()):
    if i < 15:  # Only freeze first 15 layers
        param.requires_grad = False

num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(train_dataset.classes))
model = model.to(device)

# Optimizer and loss
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("🏃‍♂️ Starting FAST training (1 epoch for quick results)...")

# Train for just 1 epoch to get immediate results
model.train()
running_loss = 0.0
batch_count = 0

for images, labels in train_loader:
    images, labels = images.to(device), labels.to(device)
    
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    
    running_loss += loss.item()
    batch_count += 1
    
    # Print progress every 50 batches
    if batch_count % 50 == 0:
        print(f"  Batch {batch_count}/{len(train_loader)} - Loss: {loss.item():.4f}")
    
    # Stop after 200 batches for quick results
    if batch_count >= 200:
        print("⚡ Quick training complete! (Limited to 200 batches for speed)")
        break

# Quick evaluation
print("📊 Testing model...")
model.eval()
correct = 0
total = 0

with torch.no_grad():
    test_count = 0
    for images, labels in test_loader:
        if test_count >= 50:  # Test on first 50 batches only
            break
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        test_count += 1

accuracy = 100 * correct / total
print(f"🎯 Quick test accuracy: {accuracy:.2f}% (on limited test set)")

# Save the model
save_dir = os.path.join(base_dir, 'checkpoints')
os.makedirs(save_dir, exist_ok=True)
model_path = os.path.join(save_dir, 'resnet18_animals10.pth')
torch.save(model.state_dict(), model_path)
print(f"💾 Fast-trained model saved to: {model_path}")
print("✅ Now try the prediction script - it should be much more accurate!")