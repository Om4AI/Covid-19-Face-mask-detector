import streamlit as st
import cv2
import os
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np

st.set_page_config(
     page_title="LIVE FACE MASK DETECTOR",
     page_icon="😷",
     layout="wide"
)
st.title("COVID-19 Live Face Mask Detector")


import warnings

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()


# Title & Headers
st.subheader("AI based Face Mask Detector\n\n\tMake a prediction using TensorFlow & Keras")
# Expander & Description
with st.expander("Project Description"):
    st.write("This Face Mask Detector is developed by Om Mule (All rights reserved).")
    st.write("From the past year there has been the life threatning pandemic of COVID-19 which has led to a lot of deaths. Face Masks are very essential which can prevent the spread of pandemic to a great extent.")
    st.write("This project is specially developed to be detect if a person is wearing Face mask or not.")


# run = st.checkbox('Run')
FRAME_WINDOW = st.image([])
video_capture = cv2.VideoCapture(0)

cascPath = os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
model = load_model("MobileNetV2-facemask.h5")

left, right = st.columns(2)
turn_on = left.button("TURN ON CAMERA")
off = right.button("TURN OFF CAMERA")
if off: turn_on = False

while turn_on:
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,
                                         scaleFactor=1.1,
                                         minNeighbors=5,
                                         minSize=(60, 60),
                                         flags=cv2.CASCADE_SCALE_IMAGE)
    faces_list=[]
    preds=[]
    for (x, y, w, h) in faces:
        face_frame = frame[y:y+h,x:x+w]
        face_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
        face_frame = cv2.resize(face_frame, (160,160))
        face_frame = img_to_array(face_frame)
        face_frame = np.expand_dims(face_frame, axis=0)
        face_frame =  np.vstack([face_frame])
        faces_list.append(face_frame)
        if len(faces_list)>0:
            preds = model.predict(faces_list, batch_size =10)
        label = "No Mask" if preds[0] > 0.5  else "Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        if (label == "No Mask"):
            perc = float (preds[0]*100)
        else: perc = float ((1-preds[0])*100)
        label = "{}: {:.2f}%".format(label, perc)
        cv2.putText(frame, label, (x, y- 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
 
        cv2.rectangle(frame, (x, y), (x + w, y + h),color, 2)
        # Display the resulting frame
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame)
video_capture.release()
cv2.destroyAllWindows()