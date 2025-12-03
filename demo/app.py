import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# --- 1. SETUP ---
st.title("📱 Phone vs 👛 Wallet Classifier")
st.write("Apni photo upload karo, mera AI batayega wo kya hai!")

# Classes
classes = ['Phone', 'Wallet']

# --- 2. LOAD MODEL (Cached) ---
# @st.cache_resource lagane se model baar-baar load nahi hoga (Fast rahega)
@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    # Path dhyan se check karna!
    model.load_state_dict(torch.load('models/custom_train.pth', map_location='cpu'))
    model.eval()
    return model

try:
    model = load_model()
    st.success("Brain (Model) Loaded Successfully! 🧠")
except:
    st.error("Model file nahi mili! Check path: 'models/custom_train.pth'")

# --- 3. PREDICTION LOGIC ---
def predict(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        conf, pred = torch.max(probs, 1)
        
    return classes[pred.item()], conf.item()

# --- 4. UI INTERFACE ---
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Image dikhao
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    st.write("Thinking... 🤔")
    
    # Predict karo
    label, score = predict(image)
    
    # Result dikhao
    if label == 'Phone':
        st.header(f"Yeh ek 📱 **{label}** hai!")
    else:
        st.header(f"Yeh ek 👛 **{label}** hai!")
        
    st.info(f"Confidence: {score*100:.2f}%")