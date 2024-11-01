import streamlit as st
import torch
import torch.nn as nn 
from torchvision import transforms
from PIL import Image
import numpy as np


st.markdown("<h2 style='text-align: center; color: blue;'>Digit Prediction App</h2>", unsafe_allow_html=True)


st.markdown("""
### How It Works
Once you upload an image, the model processes it, scales it down to match the 28x28 pixel resolution used in MNIST, and then analyzes the pixel patterns to classify which digit it represents. This is achieved by leveraging a deep learning model that has been trained to recognize subtle features in handwritten numbers, making it highly accurate at identifying digits even with varied handwriting styles.
""")

class MyConvBlock(nn.Module):
    def __init__(self):
        super(MyConvBlock, self).__init__()
        self.model = nn.Sequential(
        nn.Conv2d(1, 32,3),
        nn.ReLU(),
        nn.MaxPool2d(2,2),
            
        nn.Conv2d(32, 64,3),
        nn.ReLU(),
        nn.MaxPool2d(2,2),
            
        nn.Conv2d(64,128,3),
        nn.ReLU(),
        nn.MaxPool2d(2,2),
        
        nn.Flatten(),
            
        nn.Linear(128,64),
        nn.Linear(64,10)
        
        )
        

    def forward(self, x):
        return self.model(x)
    

model = MyConvBlock()
model.load_state_dict(torch.load("trained_model.pth"))
model.eval()  # Set model to evaluation mode



uploaded_file = st.file_uploader("Upload a digit image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to 28x28 pixels
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    img_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Make prediction
    with torch.no_grad():
        output = model(img_tensor)
        prediction = output.argmax().item()

    # Display the result
    st.write(f"Predicted Digit: {prediction}")

