import os
import cv2
import logging
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image
from img2vec_pytorch import Img2Vec
import joblib
import numpy as np

app = Flask(__name__)

# Configurations
UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
MAX_IMAGE_SIZE = (64, 64)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

try:
    model = joblib.load('svm_flower_classifier.pkl')
except Exception as e:
    logging.error(f"Error loading model: {e}")
    model = None

try:
    img2vec = Img2Vec(model='resnet-18')
except Exception as e:
    logging.error(f"Error initializing Img2Vec: {e}")
    img2vec = None

def process_image(file):
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
        file.save(filepath)
        
        image = Image.open(filepath)
        image = image.resize(MAX_IMAGE_SIZE)
        return np.array(image), filepath
    except Exception as e:
        logging.error(f"Error processing image: {e}")
        return None, None

def extract_features(image):
    try:
        image_pil = Image.fromarray(image)
        features = img2vec.get_vec(image_pil, tensor=False)
        return features
    except Exception as e:
        logging.error(f"Error extracting features: {e}")
        return None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        logging.error("No image part in the request")
        return redirect(request.url)
    
    file = request.files['image']
    if file.filename == '':
        logging.error("No file selected for uploading")
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        image, filepath = process_image(file)
        if image is not None:
            features = extract_features(image)
            if features is not None:
                features = features.reshape(1, -1)
                try:
                    prediction = model.predict(features)
                    return redirect(url_for('output', value=int(prediction[0]), image_path=filepath))
                except Exception as e:
                    logging.error(f"Error making prediction: {e}")
                    return redirect(url_for('index'))
        else:
            logging.error("Error reading the image")
    
    return redirect(url_for('index'))

@app.route('/output')
def output():
    value = request.args.get('value', default=-1, type=int)
    image_path = request.args.get('image_path')
    return render_template('output.html', value=value, image_path=image_path)

if __name__ == '__main__':
    app.run(debug=False)
