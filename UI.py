import os
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import uvicorn
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io

app = FastAPI()

# Load the ensemble model
ensemble_model = load_model('ensemble_model1.h5')

# Function to get class names from dataset directory
def get_class_names(dataset_dir):
    entries = os.listdir(dataset_dir)
    class_names = [entry for entry in entries if os.path.isdir(os.path.join(dataset_dir, entry))]
    class_names.sort()
    return class_names

# Load class names from the dataset directory
dataset_directory = 'path to your valid dataset'  
class_names = get_class_names(dataset_directory)

# Set up templates
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def main(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict/", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    # Read image file
    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert('RGB')
    except Exception as e:
        return templates.TemplateResponse("error.html", {"request": request, "message": f"Invalid image file: {e}"})

    # Preprocess the image
    img = img.resize((256, 256))  
    img_array = np.array(img) / 255.0  
    img_array = np.expand_dims(img_array, axis=0)  

    # Use the ensemble model
    selected_model = ensemble_model

    # Make prediction
    predictions = selected_model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions)

    predicted_class = class_names[predicted_class_index]

    result = {
        "class": predicted_class,
        "confidence": float(confidence)
    }

    return templates.TemplateResponse("result.html", {"request": request, "result": result})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

