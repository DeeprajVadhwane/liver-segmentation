import os
import numpy as np
import nibabel as nib
import tensorflow as tf
from skimage.transform import resize
from flask import Flask, request, render_template, redirect, url_for

app = Flask(__name__)

# Create the temporary directory if it doesn't exist
if not os.path.exists('temp'):
    os.makedirs('temp')

# Load your pre-trained ResNet model
model = tf.keras.models.load_model('resnet_model.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(url_for('home'))
    
    file = request.files['file']
    
    if file.filename == '':
        return redirect(url_for('home'))
    
    if file:
        # Save the uploaded file to a temporary location
        file_path = os.path.join('temp', file.filename)
        file.save(file_path)
        
        # Preprocess the uploaded .nii file
        img_data = read_nii(file_path)
        
        # Further preprocess the image data for the model
        img_data = preprocess_image(img_data)
        
        # Make predictions
        prediction = model.predict(img_data)
        
        # Interpret the prediction
        result = interpret_prediction(prediction)
        
        # Remove the temporary file
        os.remove(file_path)
        
        return render_template('result.html', result=result)
    return redirect(url_for('home'))

def read_nii(filepath):
    ct_scan = nib.load(filepath)
    array = ct_scan.get_fdata()
    array = np.rot90(np.array(array))
    return array

def preprocess_image(img_data):
    # Assume the model expects (1, 224, 224, 3) input shape
    if len(img_data.shape) == 3:  # 3D NIfTI image (single channel)
        img_data = img_data[..., np.newaxis]

    # Resize to the model input size, e.g., (224, 224, 224, 1)
    img_data = resize(img_data, (224, 224, 224, 1), mode='constant', preserve_range=True)
    img_data = np.expand_dims(img_data, axis=0)  # Add batch dimension

    # Normalize the image data
    img_data = img_data / np.max(img_data)
    
    return img_data

def interpret_prediction(prediction):
    # Implement your prediction interpretation logic here
    return "Tumor Detected" if prediction[0] > 0.5 else "No Tumor Detected"

if __name__ == '__main__':
    app.run(debug=True)
