import joblib
import numpy as np
from PIL import Image
from skimage.feature import hog
from skimage import color
import tkinter as tk
from tkinter import filedialog

# Function to process the image
def process_image(image_path):
    img = Image.open(image_path).convert('L')  # Convert image to grayscale
    img = img.resize((64, 64))  # Resize to match training size
    img_array = np.array(img)
    features = hog(img_array, block_norm='L2-Hys', pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
    return features.reshape(1, -1)  # Reshape for prediction

# Load the trained model
model = joblib.load('svm_model.pkl')

# Create a file dialog to choose an image
root = tk.Tk()
root.withdraw()  # Hide the main window
file_path = filedialog.askopenfilename(title="Select an image", filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.tiff;*.gif")])

if file_path:
    # Process the selected image
    image_features = process_image(file_path)

    # Make prediction
    prediction = model.predict(image_features)

    # Output result
    label = 'Dog' if prediction[0] == 1 else 'Cat'
    print(f'The image is classified as: {label}')
else:
    print("No file selected.")
