import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import cv2  # OpenCV (Computer Vision)
import numpy as np
import os

# --- 1. MODEL ARCHITECTURE (Wahi purana dost) ---
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Linear(64 * 7 * 7, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

# --- 2. LOAD MODEL ---
model = CNN()
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, '../models/mnist_cnn_v1.pth')

try:
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    print("✅ Model Loaded! Camera starting...")
except:
    print("❌ Model nahi mila! Path check kar.")
    exit()

# --- 3. HELPER: PREPROCESS FRAME ---
def preprocess_frame(frame):
    # A. Grayscale (Color hatao)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # B. Thresholding (Important!)
    # Jo bhi thoda sa bhi dark hai usko BLACK kar do, baki WHITE.
    # Isse background noise hat jata hai.
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    
    return thresh

# --- 4. WEBCAM LOOP 🔄 ---
# 0 ka matlab default webcam
cap = cv2.VideoCapture(0)

# Screen pe ek box banayenge jahan tu number dikhayega
x1, y1, x2, y2 = 200, 200, 450, 450

while True:
    # 1. Frame Read karo
    ret, frame = cap.read()
    if not ret: break
    
    # Mirror effect (Selfie mode)
    frame = cv2.flip(frame, 1)
    
    # 2. Extract Region of Interest (ROI) - Wo chhota box
    roi = frame[y1:y2, x1:x2]
    
    # 3. Process that box (Black & White conversion)
    processed_roi = preprocess_frame(roi)
    
    # 4. PyTorch Prediction
    # OpenCV image (Numpy) -> PIL Image -> Tensor
    pil_img = Image.fromarray(processed_roi)
    pil_img = pil_img.resize((28, 28))
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    img_tensor = transform(pil_img).view(1, 1, 28, 28)
    
    with torch.no_grad():
        output = model(img_tensor)
        prob = F.softmax(output, dim=1)
        conf, pred = torch.max(prob, 1)
        
    # 5. Display Result on Screen
    prediction_text = f"Digit: {pred.item()}"
    conf_text = f"Conf: {conf.item()*100:.1f}%"
    
    # Box banao
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Text likho
    cv2.putText(frame, prediction_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, conf_text, (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Chhota window dikhao ki model ko kya dikh raha hai (Black & White view)
    cv2.imshow("What Model Sees", processed_roi)
    cv2.imshow("Webcam Feed", frame)
    
    # 'q' dabane se band hoga
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()