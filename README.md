# Face_mask_detector

## During these tough times of COVID 19; face masks have become very important and essential

**This model is developed with Keras and TensorFlow can be used to detect whether people are wearing a mask or not.**

It uses the **MobileNetV2** for Transfer Learning and uses the dataset from Kaggle to train the images and correctly classify images as **Wearing Mask / Not Wearing Mask.** **Kaggle dataset link:**    [Kaggle_Face-Mask_Dataset](https://www.kaggle.com/ashishjangra27/face-mask-12k-images-dataset)

Also there is an upload option provided so that we can check how the model performs with the new images from our desktop that we feed it.

<p align="center"><img src="My_mask_test.PNG" height="400" width="400"></p>

## *Check on your own images:*
#### 1. Take an image from your Webcam.
#### 2. Run and train the model (I generally use [Google Colab](https://colab.research.google.com/)).
#### 3. Run the upload cell from the code, it will prompt you to upload a photo from your device; upload the photo which you took in Step 1 & get the results.
#### 4. (Update) Code for Realtime Checking using the model has been included in the repository now. For more reference regarding *Saving and Loading Models*, view the official documentation: [TensorFlow: Load and Save models](https://www.tensorflow.org/tutorials/keras/save_and_load)
#### 5. ***Real time face mask detection*** has been included in the repository now. I have used OpenCV (Haarcascades) to take only the face area of the person and feed it to the model that would be a saved H5 file.


### Check Face Mask Detector using your own Webcam: Run the [Face Mask Detector](https://github.com/Om4AI/Covid-Face-mask-detector/blob/main/Try_Face_Mask_detector.py)

### Code for a Flask App has also been included to get outputs from a web app in local host.
