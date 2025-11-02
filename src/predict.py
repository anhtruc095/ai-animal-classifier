import torch
from torchvision import models, transforms
from PIL import Image
import argparse
import os
import matplotlib.pyplot as plt

# 1️⃣ Danh sách class giống lúc train
classes = ['cat', 'butterfly', 'dog', 'sheep', 'spider', 'chicken', 'horse', 'squirrel', 'cow', 'elephant']

# 2️⃣ Nhận đường dẫn ảnh
parser = argparse.ArgumentParser()
parser.add_argument('--image', type=str, required=True, help='Path to image')
args = parser.parse_args()
image_path = args.image

# 3️⃣ Load model đã train - Match the training architecture
model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
# Use same architecture as training (simple linear layer)
model.fc = torch.nn.Linear(num_ftrs, len(classes))

# Fix path and check if model exists
import os
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(base_dir, 'checkpoints', 'resnet18_animals10.pth')

if not os.path.exists(model_path):
    print(f"Model file not found at: {model_path}")
    print("Please train the model first by running: python src/train.py")
    exit(1)

model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# 4️⃣ Chuẩn bị transform giống lúc train
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 5️⃣ Mở ảnh
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Không tìm thấy ảnh: {image_path}")

image = Image.open(image_path).convert('RGB')
input_tensor = transform(image).unsqueeze(0)

# 6️⃣ Dự đoán
with torch.no_grad():
    outputs = model(input_tensor)
    _, predicted = outputs.max(1)
    label = classes[predicted.item()]

print(f"Predicted: {label}")

# 7️⃣ Hiển thị ảnh kèm nhãn dự đoán
plt.imshow(image)
plt.title(f"Predicted: {label}", fontsize=16, color='green')
plt.axis('off')
plt.show()