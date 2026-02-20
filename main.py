import os
from pathlib import Path
from face_processor import align_face
from dataset import LFWSubset, transform
from torch.utils.data import DataLoader

lfw_path = "/kaggle/input/lfw-dataset/lfw-deepfunneled/lfw-deepfunneled"
min_images = 10  # only keep people with 10+ photos

valid_people = []
for person in os.listdir(lfw_path):
    person_path = os.path.join(lfw_path, person)
    # Skip if not a directory (e.g., .DS_Store files)
    if not os.path.isdir(person_path):
        continue
    images = os.listdir(person_path)
    if len(images) >= min_images:
        valid_people.append((person, len(images)))

print(f"Valid identities: {len(valid_people)}")
# Expect ~158 people, ~2500 images total

# Prepare dataset
image_paths = []
labels = []
label_to_idx = {person: idx for idx, (person, _) in enumerate(valid_people)}

for person, _ in valid_people:
    person_path = os.path.join(lfw_path, person)
    for img_name in os.listdir(person_path):
        if img_name.endswith(('.jpg', '.png', '.jpeg')):
            img_path = os.path.join(person_path, img_name)
            image_paths.append(img_path)
            labels.append(label_to_idx[person])

print(f"Total images: {len(image_paths)}")
print(f"Total classes: {len(label_to_idx)}")

# Create dataset and dataloader
dataset = LFWSubset(image_paths, labels, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

print(f"Dataset created with {len(dataset)} images")