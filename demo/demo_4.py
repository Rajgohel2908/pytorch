from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# 1. Load Model
model = YOLO('yolov8n.pt') 

# 2. Predict
# Hum 'bus.jpg' internet se utha rahe hain
print("Detecting objects... 🚌")
results = model('download.jpg')

# 3. Magic Step: Get Image with Boxes 🎁
# plot() function image ke upar boxes draw karke numpy array deta hai
res_plotted = results[0].plot()

# 4. Color Fix (OpenCV vs Matplotlib) 🎨
# OpenCV Blue-Green-Red (BGR) use karta hai.
# Matplotlib Red-Green-Blue (RGB) use karta hai.
# Agar convert nahi kiya, toh Bus "Neeli" dikhegi.
res_plotted = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)

# 5. Show using Matplotlib (The "Stay" command)
plt.figure(figsize=(10, 10))
plt.imshow(res_plotted)
plt.axis('off') # Numbers hatao side ke
plt.title("YOLO Detection Result")
print("Image screen pe hai. Band karne ke liye Window close karo.")
plt.show() # Yeh line code ko rok ke rakhegi jab tak tu window band na kare