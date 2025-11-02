import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import os

print("🎯 STARTING HIGH ACCURACY TRAINING")
print("This will train for real accuracy on your animal dataset")
print("Expected final accuracy: 85-95%")
print("-" * 50)

# Paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
train_dir = os.path.join(base_dir, 'data', 'train')
test_dir = os.path.join(base_dir, 'data', 'test')

print("Loading dataset...")

# Simple but effective transforms
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

print(f"Training images: {len(train_dataset)}")
print(f"Test images: {len(test_dataset)}")
print(f"Classes: {train_dataset.classes}")

# Model setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training on: {device}")

# Use ResNet18 with proper transfer learning
model = models.resnet18(pretrained=True)

# Fine-tune last layers only for speed + accuracy
for param in model.parameters():
    param.requires_grad = False

# Unfreeze last layer block for learning
for param in model.layer4.parameters():
    param.requires_grad = True

# Replace final layer
model.fc = nn.Linear(model.fc.in_features, len(train_dataset.classes))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Starting training...")
print("-" * 50)

# Train for 5 epochs - good balance of speed vs accuracy
best_accuracy = 0

for epoch in range(5):
    print(f"\nEPOCH {epoch + 1}/5")
    
    # Training
    model.train()
    total_loss = 0
    batch_count = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        batch_count += 1
        
        if batch_idx % 200 == 0:
            print(f"  Batch {batch_idx}/{len(train_loader)}: Loss = {loss.item():.3f}")
    
    avg_loss = total_loss / batch_count
    
    # Testing
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f"  Loss: {avg_loss:.3f} | Accuracy: {accuracy:.1f}%")
    
    # Save best model
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        save_dir = os.path.join(base_dir, 'checkpoints')
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, 'resnet18_animals10.pth')
        torch.save(model.state_dict(), model_path)
        print(f"  *** NEW BEST MODEL SAVED: {accuracy:.1f}% ***")

print("-" * 50)
print(f"🎉 TRAINING COMPLETE!")
print(f"🏆 Best accuracy: {best_accuracy:.1f}%")
print(f"💾 Model saved to: checkpoints/resnet18_animals10.pth")
print("✅ Ready for high-accuracy predictions!")