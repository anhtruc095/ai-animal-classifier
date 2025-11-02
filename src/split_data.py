import os
import shutil
import random

# Path raw dataset
raw_dir = './data/Animals-10/raw-img'
train_dir = './data/train'
test_dir = './data/test'
split_ratio = 0.8  # 80% train, 20% test

# Mapping class name nếu muốn dùng tiếng Anh
class_map = {
    'cane': 'dog',
    'cavallo': 'horse',
    'elefante': 'elephant',
    'farfalla': 'butterfly',
    'gallina': 'chicken',
    'gatto': 'cat',
    'mucca': 'cow',
    'pecora': 'sheep',
    'ragno': 'spider',
    'scoiattolo': 'squirrel'
}

# Tạo folder train/test
for folder in [train_dir, test_dir]:
    os.makedirs(folder, exist_ok=True)
    for c in class_map.values():
        os.makedirs(os.path.join(folder, c), exist_ok=True)

# Chia dữ liệu
for class_it, class_en in class_map.items():
    imgs = os.listdir(os.path.join(raw_dir, class_it))
    random.shuffle(imgs)
    split = int(len(imgs)*split_ratio)
    train_imgs = imgs[:split]
    test_imgs = imgs[split:]

    for img in train_imgs:
        shutil.copy(os.path.join(raw_dir, class_it, img),
                    os.path.join(train_dir, class_en, img))
    for img in test_imgs:
        shutil.copy(os.path.join(raw_dir, class_it, img),
                    os.path.join(test_dir, class_en, img))

print("✅ Done splitting dataset!")