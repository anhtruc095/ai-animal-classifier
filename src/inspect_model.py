import torch
import os

# Load and inspect the saved model
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(base_dir, 'checkpoints', 'resnet18_animals10.pth')

print("🔍 Inspecting saved model architecture...")
state_dict = torch.load(model_path, map_location='cpu')

print("Keys related to 'fc' layer:")
fc_keys = [key for key in state_dict.keys() if 'fc' in key]
for key in fc_keys:
    print(f"  {key}: {state_dict[key].shape}")

print(f"\nTotal keys: {len(state_dict)}")
print("Last 10 keys:")
all_keys = list(state_dict.keys())
for key in all_keys[-10:]:
    print(f"  {key}")