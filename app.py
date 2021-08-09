from __future__ import division, print_function
# General libraries
import sys
import os
import glob
import re
import numpy as np

# Keras
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Defining a flask app
app = Flask(__name__)

model_path = "MobileNetV2-facemask.h5"

# Loading the trained model
model = keras.models.load_model(model_path)
# model.__make_predict_function()

print('Model loaded. Check http://127.0.0.1:5000/')

# model predict function
def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(160,160))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x, mode='caffe')
    
    preds = model.predict(x)
    return preds

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def upload():
    if request.method == 'POST':
        #Get the file from post request
        f = request.files['file']
        
        # Save file to /uploads 
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        
        # Make predictions
        preds = model_predict(file_path, model)
        if (preds[0]>0.5):result = "No Mask"
        else: result = "Mask"
        return result
    return None

if __name__=='__main__':
    app.run(debug=True)
