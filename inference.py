import torch
from mobilefacenet import MobileFaceNet

# Load trained model
model = MobileFaceNet(embedding_size=512)
checkpoint = torch.load('./checkpoints/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Extract embedding from a face image
import torchvision.transforms as transforms
from PIL import Image

transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

image = Image.open('face.jpg').convert('RGB')
image_tensor = transform(image).unsqueeze(0)

with torch.no_grad():
    embedding = model(image_tensor)
    embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)

print(f"Embedding shape: {embedding.shape}")  # (1, 512)