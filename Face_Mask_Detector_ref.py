import streamlit as st
import tensorflow as tf
from tensorflow import keras
import os
from PIL import Image
import numpy as np
import io
# from cv2 import resize
from io import BytesIO
import base64
import skimage.transform as skimg

# Keras utils
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions


import warnings

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()


# Title & Headers
st.title("COVID-19 Face Mask Detector")
st.subheader("AI based Face Mask Detector\n\n\tMake a prediction using TensorFlow & Keras")
# Expander & Description
with st.expander("Project Description"):
    st.write("This Face Mask Detector is developed by Om Mule (All rights reserved).")
    st.write("From the past year there has been the life threatning pandemic of COVID-19 which has led to a lot of deaths. Face Masks are very essential which can prevent the spread of pandemic to a great extent.")
    st.write("This project is specially developed to be detect if a person is wearing Face mask or not.")

# Get Image & Layout items
left, right = st.columns(2)

img = left.file_uploader("Image to Test")
if (img):right.image(img, caption="Uploaded Image")
else:
    st.success("Please Upload an Image")

st.sidebar.map()


# Get the image uploaded by User

if (img):
    data = img.read()
    img = Image.open(io.BytesIO(data))
    array = image.img_to_array(img)
    # img.save("User_Image.jpeg")
# img_path = "User_Image.jpeg"

# Array has the image converted to Numpy Array
# Resize the array to (160,160)
if (img):array = skimg.resize(array, (160,160,3))

# Face Mask Model 
model_path = "MobileNetV2-facemask.h5"
model = keras.models.load_model(model_path)

# Model predict function
def model_predict(model):
    if (img):
        x = array
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x, mode='caffe')
        preds = model.predict(x)
        return preds
#     else: st.success("Please Upload an Image")

# Calling prediction function
preds = model_predict(model)
if (img and preds[0]>0.5):
    left.error("No Mask Detected")



