import os
from dataset import LFWSubset, transform
from torch.utils.data import DataLoader
from face_processor import align_face

# Test 1: Check dataset structure
print("=" * 50)
print("TEST 1: Dataset Structure")
print("=" * 50)

lfw_path = "./lfw-deepfunneled/lfw-deepfunneled"
min_images = 10

valid_people = []
for person in os.listdir(lfw_path):
    person_path = os.path.join(lfw_path, person)
    if not os.path.isdir(person_path):
        continue
    images = os.listdir(person_path)
    if len(images) >= min_images:
        valid_people.append((person, len(images)))

print(f"✓ Valid identities: {len(valid_people)}")
print(f"✓ Top 5 people: {valid_people[:5]}")

# Test 2: Check dataset loading
print("\n" + "=" * 50)
print("TEST 2: Dataset Loading")
print("=" * 50)

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

print(f"✓ Total images: {len(image_paths)}")
print(f"✓ Total classes: {len(label_to_idx)}")

dataset = LFWSubset(image_paths, labels, transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

print(f"✓ Dataset created with {len(dataset)} images")

# Test 3: Load one batch
print("\n" + "=" * 50)
print("TEST 3: Load One Batch")
print("=" * 50)

batch_images, batch_labels = next(iter(dataloader))
print(f"✓ Batch shape: {batch_images.shape}")
print(f"✓ Labels shape: {batch_labels.shape}")
print(f"✓ Image tensor range: [{batch_images.min():.2f}, {batch_images.max():.2f}]")

# Test 4: Face detection on one image
print("\n" + "=" * 50)
print("TEST 4: Face Detection")
print("=" * 50)

test_image = image_paths[0]
print(f"Testing on: {test_image}")
result = align_face(test_image)
if result:
    normed_emb, emb = result
    print(f"✓ Face detected!")
    print(f"✓ Normalized embedding shape: {normed_emb.shape}")
    print(f"✓ Embedding shape: {emb.shape}")
else:
    print("✗ No face detected")

print("\n" + "=" * 50)
print("ALL TESTS COMPLETED!")
print("=" * 50)