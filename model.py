import numpy as np
from PIL import Image
import io
import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K

# Define Dice coefficient as a metric
def dice_coefficient(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    union = K.sum(y_true, axis=-1) + K.sum(y_pred, axis=-1)
    return (2. * intersection + smooth) / (union + smooth)

# Function to preprocess the image
def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('L')
    image = image.resize((128, 128))
    image = np.array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=[0, -1])
    return image

# Load the Keras model
def load_resnet_model(model_path='resnet_model.h5'):
    return load_model(model_path, custom_objects={'dice_coefficient': dice_coefficient})