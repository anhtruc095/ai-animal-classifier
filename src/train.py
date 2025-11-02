import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Biến config
import os
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
train_dir = os.path.join(base_dir, 'data', 'train')
test_dir = os.path.join(base_dir, 'data', 'test')
batch_size = 16  # Reduced for faster training

# Transform (data augmentation)
train_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

# Dataset
print(f"Looking for training data at: {train_dir}")
print(f"Looking for test data at: {test_dir}")
import os
print(f"Current working directory: {os.getcwd()}")
print(f"Train dir exists: {os.path.exists(train_dir)}")
print(f"Test dir exists: {os.path.exists(test_dir)}")
if os.path.exists(train_dir):
    print(f"Contents of train dir: {os.listdir(train_dir)}")

train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
import torch.nn as nn
from torchvision import models

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load pretrained ResNet18 (smaller, faster)
model = models.resnet18(pretrained=True)

# Freeze layer đầu
for param in model.parameters():
    param.requires_grad = False

# Thay FC layer cho 10 class
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 10)
model = model.to(device)

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

num_epochs = 3  # Reduced for faster training
train_losses, test_accuracies = [], []

print(f"Starting training for {num_epochs} epochs...")
print(f"Training on device: {device}")
print(f"Number of training batches: {len(train_loader)}")
print(f"Number of test batches: {len(test_loader)}")

for epoch in range(num_epochs):
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

        running_loss += loss.item() * images.size(0)
        batch_count += 1
        
        # Print progress every 10 batches
        if batch_count % 10 == 0:
            print(f"  Batch {batch_count}/{len(train_loader)} - Loss: {loss.item():.4f}")
    
    scheduler.step()
    epoch_loss = running_loss / len(train_loader.dataset)
    train_losses.append(epoch_loss)
    
    # Evaluation
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    test_accuracies.append(acc)
    print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {epoch_loss:.4f} - Test Acc: {acc:.4f}")

# Plotpy
plt.plot(range(1,num_epochs+1), train_losses, label='Loss')
plt.plot(range(1,num_epochs+1), test_accuracies, label='Accuracy')
plt.legend()
plt.show()

# Save model with correct path and name
save_dir = os.path.join(base_dir, 'checkpoints')
os.makedirs(save_dir, exist_ok=True)
model_path = os.path.join(save_dir, 'resnet18_animals10.pth')
torch.save(model.state_dict(), model_path)
print(f"Model saved to: {model_path}")