import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os

# Function to get class_names from dataset directory
def get_class_names(dataset_dir):
    entries = os.listdir(dataset_dir)
    class_names = [entry for entry in entries if os.path.isdir(os.path.join(dataset_dir, entry))]
    class_names.sort()
    return class_names

# Load class names from the dataset directory
dataset_directory = 'PATH TO VALID DATASET'  # Replace with the actual path to your dataset directory
class_names = get_class_names(dataset_directory)

# Load the ensemble model
ensemble_model = load_model('ensemble_model1.h5')

# Set page config (optional)
st.set_page_config(
    page_title="Bird Image Classifier",
    page_icon="🦉",
    layout="centered",
)

# Title and instructions
st.title("Bird Image Classifier 🦉")
st.write("Upload a bird image, and the classifier will predict the species using the ensemble model.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image file
    try:
        img = Image.open(uploaded_file).convert('RGB')
    except Exception as e:
        st.error(f"Error loading image: {e}")
    else:
        # Display the uploaded image
        st.image(img, caption='Uploaded Image', use_column_width=True)

        # Predict button
        if st.button('Classify Image'):
            with st.spinner('Classifying...'):
                # Preprocess the image
                img_resized = img.resize((256, 256))
                img_array = np.array(img_resized) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                # Make prediction
                predictions = ensemble_model.predict(img_array)
                predicted_class_index = np.argmax(predictions, axis=1)[0]
                confidence = np.max(predictions)

                predicted_class = class_names[predicted_class_index]

                # Display result
                st.success(f"Prediction: **{predicted_class}**")
                st.write(f"Confidence: **{confidence * 100:.2f}%**")
