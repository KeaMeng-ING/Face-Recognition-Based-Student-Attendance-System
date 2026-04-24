import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms, models
import torch.nn as nn
import numpy as np

# Use the same model architecture as app.py
class FFTBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AvgPool2d(kernel_size=3, stride=3),
        )
    def forward(self, x): return self.net(x).flatten(1)

class AntiSpoofNet(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.mobilenet_v2(weights=None)
        self.rgb_branch = base.features
        self.rgb_pool   = nn.AdaptiveAvgPool2d(1)
        self.fft_branch = FFTBranch()
        self.classifier = nn.Sequential(
            nn.Linear(1280 + 2304, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Dropout(0.5), nn.Linear(512, 128), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(128, 1),
        )
    def forward(self, rgb, fft):
        r = self.rgb_pool(self.rgb_branch(rgb)).flatten(1)
        f = self.fft_branch(fft)
        return self.classifier(torch.cat([r, f], dim=1)).squeeze(1)

def get_fft(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray = cv2.resize(gray, (80, 80))
    mag  = np.log(np.abs(np.fft.fftshift(np.fft.fft2(gray))) + 1e-8)
    mag  = (mag - mag.mean()) / (mag.std() + 1e-8)
    return torch.from_numpy(mag).unsqueeze(0).unsqueeze(0)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AntiSpoofNet().to(device)
    ckpt = torch.load("./checkpoints/best_model_antispoofing.pth", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    trans = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((80, 80)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    cap = cv2.VideoCapture(0)
    print("\n🔬 Anti-Spoof Debugger")
    print("Shows RAW scores. 1.0 = REAL, 0.0 = SPOOF\n")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # Simple center crop for speed
        h, w = frame.shape[:2]
        s = min(h, w) // 2
        crop = frame[h//2-s:h//2+s, w//2-s:w//2+s]
        crop_disp = cv2.resize(crop, (200, 200))

        rgb_t = trans(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)
        fft_t = get_fft(crop).to(device)

        with torch.no_grad():
            raw = torch.sigmoid(model(rgb_t, fft_t)).item()

        color = (0, 255, 0) if raw > 0.5 else (0, 0, 255)
        cv2.putText(frame, f"RAW SCORE: {raw:.4f}", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        
        cv2.imshow("Debug Spoof", frame)
        print(f"\rCurrent Raw Score: {raw:.4f} | {'REAL' if raw > 0.5 else 'SPOOF'}", end="")

        if cv2.waitKey(1) & 0xFF == 27: break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
