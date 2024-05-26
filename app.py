from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

app = Flask(__name__)

# Load your trained model
model = load_model('fcn_model.h5')

# Recompile the model to ensure metrics are properly initialized
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.Precision()])

# Define a function to process and predict the image
def predict_tumor(image_path):
    img = load_img(image_path, target_size=(224, 224))  # Resize to your model's input size
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize to [0, 1]

    prediction = model.predict(img_array)
    print(f"Prediction shape: {prediction.shape}, Prediction: {prediction}")

    # Check if the predicted mask has any non-zero values
    has_tumor = np.any(prediction > 0.5)

    return 'Yes' if has_tumor else 'No'

# Define the routes for the web app
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return render_template('index.html', prediction='No file part')
        
        file = request.files['file']
        
        # If user does not select file, browser also submits an empty part without filename
        if file.filename == '':
            return render_template('index.html', prediction='No selected file')
        
        if file:
            file_path = os.path.join('uploads', file.filename)
            file.save(file_path)
            
            # Predict the uploaded image
            prediction = predict_tumor(file_path)
            return render_template('index.html', prediction=prediction)
    
    return render_template('index.html', prediction='')

if __name__ == "__main__":
    # Ensure the upload folder exists
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)
