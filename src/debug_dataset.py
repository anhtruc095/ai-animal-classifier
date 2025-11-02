import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

print("Starting debug...")

# Get paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
train_dir = os.path.join(base_dir, 'data', 'train')
print(f"Train directory: {train_dir}")

# Simple transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

print("Creating dataset...")
try:
    dataset = datasets.ImageFolder(train_dir, transform=transform)
    print(f"Dataset created successfully!")
    print(f"Number of classes: {len(dataset.classes)}")
    print(f"Classes: {dataset.classes}")
    print(f"Number of samples: {len(dataset)}")
    
    print("Creating DataLoader...")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
    print(f"DataLoader created successfully!")
    
    print("Testing first batch...")
    for i, (images, labels) in enumerate(dataloader):
        print(f"Batch {i}: images shape = {images.shape}, labels = {labels}")
        if i >= 2:  # Only test 3 batches
            break
    
    print("✅ Dataset loading test completed successfully!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()