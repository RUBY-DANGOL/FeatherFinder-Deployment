from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load the trained model
MODEL_PATH = "ensemble_model1.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Define allowed extensions for uploaded files
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
UPLOAD_FOLDER = "static/uploads"
DATASET_FOLDER = "D:/Birds_Final/valid"  # Folder containing subfolders for each bird class
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(img_path):
    target_size = (256, 256)  # Resize image to match model input size
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize
    return img_array

# Dynamically fetch class names from dataset folder
if os.path.exists(DATASET_FOLDER):
    class_labels = sorted([d for d in os.listdir(DATASET_FOLDER) if os.path.isdir(os.path.join(DATASET_FOLDER, d))])
else:
    class_labels = []

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict/', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('error.html', message="No file uploaded.")
    
    file = request.files['file']
    if file.filename == "":
        return render_template('error.html', message="No file selected.")
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Preprocess the image
        img_array = preprocess_image(filepath)
        
        # Make prediction
        predictions = model.predict(img_array)
        class_index = np.argmax(predictions)
        confidence = np.max(predictions)
        
        # Get predicted class name
        if class_index < len(class_labels):
            predicted_class = class_labels[class_index]
        else:
            predicted_class = "Unknown"
        
        return render_template('result.html', result={'class': predicted_class, 'confidence': confidence})
    else:
        return render_template('error.html', message="Invalid file format. Allowed formats: png, jpg, jpeg.")

if __name__ == '__main__':
    app.run(debug=True)
