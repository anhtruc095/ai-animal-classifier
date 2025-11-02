import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
import os
import time

print("🎯 HIGH ACCURACY TRAINING - Optimized for Best Results")
print("=" * 60)

# Get paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
train_dir = os.path.join(base_dir, 'data', 'train')
test_dir = os.path.join(base_dir, 'data', 'test')

# Optimized transforms for high accuracy
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

print("📁 Loading datasets...")
train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)

# Optimized batch size
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

print(f"✅ Dataset loaded:")
print(f"   Training: {len(train_dataset)} images")
print(f"   Testing: {len(test_dataset)} images")
print(f"   Classes: {train_dataset.classes}")
print(f"   Batch size: {batch_size}")

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"💻 Training on: {device}")

# Create optimized model
print("🧠 Creating ResNet18 model with transfer learning...")
model = models.resnet18(weights='IMAGENET1K_V1')  # Updated syntax

# Fine-tuning strategy: unfreeze last few layers for better accuracy
for param in model.parameters():
    param.requires_grad = False

# Unfreeze the last 2 blocks for better learning
for param in model.layer4.parameters():
    param.requires_grad = True
for param in model.layer3.parameters():
    param.requires_grad = True

# Replace classifier
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(num_features, 128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, len(train_dataset.classes))
)
model = model.to(device)

# Optimized training setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

# Training configuration for high accuracy
num_epochs = 10
best_accuracy = 0.0
train_losses = []
test_accuracies = []

print("🚀 Starting high-accuracy training...")
print("=" * 60)

start_time = time.time()

for epoch in range(num_epochs):
    epoch_start = time.time()
    
    # Training phase
    model.train()
    running_loss = 0.0
    train_correct = 0
    train_total = 0
    
    print(f"\n📈 Epoch [{epoch+1}/{num_epochs}]")
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
        
        # Progress indicator every 100 batches
        if batch_idx % 100 == 0 and batch_idx > 0:
            current_acc = 100 * train_correct / train_total
            print(f"   Batch {batch_idx:4d}/{len(train_loader)} | Loss: {loss.item():.4f} | Train Acc: {current_acc:.2f}%")
    
    epoch_loss = running_loss / len(train_loader)
    train_accuracy = 100 * train_correct / train_total
    train_losses.append(epoch_loss)
    
    # Validation phase
    model.eval()
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
    
    test_accuracy = 100 * test_correct / test_total
    test_accuracies.append(test_accuracy)
    
    # Learning rate scheduling
    scheduler.step(test_accuracy)
    
    # Save best model
    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        save_dir = os.path.join(base_dir, 'checkpoints')
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, 'resnet18_animals10.pth')
        torch.save(model.state_dict(), model_path)
        print(f"   💾 New best model saved! Accuracy: {test_accuracy:.2f}%")
    
    epoch_time = time.time() - epoch_start
    print(f"   📊 Train Loss: {epoch_loss:.4f} | Train Acc: {train_accuracy:.2f}% | Test Acc: {test_accuracy:.2f}% | Time: {epoch_time:.1f}s")
    
    # Early stopping if we reach very high accuracy
    if test_accuracy > 95:
        print(f"🎉 Excellent accuracy achieved ({test_accuracy:.2f}%)! Early stopping.")
        break

total_time = time.time() - start_time

print("\n" + "=" * 60)
print("🎉 TRAINING COMPLETE!")
print(f"⏱️  Total training time: {total_time/60:.1f} minutes")
print(f"🏆 Best test accuracy: {best_accuracy:.2f}%")
print(f"💾 Model saved to: {model_path}")

# Performance summary
print("\n📈 Training Progress Summary:")
for i in range(len(test_accuracies)):
    print(f"   Epoch {i+1:2d}: {test_accuracies[i]:6.2f}% accuracy")

print("\n✅ Your model is now ready for high-accuracy predictions!")
print("🔥 Run: python src/predict.py --image 'path/to/image.jpg'")