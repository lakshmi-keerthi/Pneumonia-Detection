from flask import Flask, render_template, request
from PIL import Image
import io
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load the trained CNN model
model = tf.keras.models.load_model('CNN_model.h5')

# Define a function to preprocess the uploaded image
def preprocess_image(image):
    # Ensure the image is in RGB mode
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Resize the image to match the model's input shape
    img = image.resize((180, 180))  # Update size to (180, 180)
    
    # Convert the image to a NumPy array and normalize pixel values
    img = np.array(img) / 255.0  # Normalize to [0, 1]
    
    # Add a batch dimension
    img = np.expand_dims(img, axis=0)
    return img

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    # Get the uploaded image from the request
    image = request.files['image'].read()
    image = Image.open(io.BytesIO(image))
    
    # Preprocess the image
    img = preprocess_image(image)
    
    # Perform classification using the loaded model
    result = model.predict(img)
    
    # Interpret the prediction result
    if result[0][0] > 0.5:
        prediction = 'Pneumonia Detected'
    else:
        prediction = 'No Pneumonia Detected'
    
    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
