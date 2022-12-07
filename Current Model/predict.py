from time import sleep
from leap_data_helper import *
import numpy as np
import sys
import ctypes
import tensorflow as tf
import os
from cv2 import imread


model = tf.keras.models.load_model(os.path.abspath(os.getcwd()) + "/model.h5")
data_dir = 'Image_Directory'
#getting the labels form data directory
labels = sorted(os.listdir(data_dir))

img = imread("letter.jpg")
prediction = model.predict(img.reshape(1,50,50,3))
char_index = np.argmax(prediction)
confidence = round(prediction[0,char_index]*100, 1)
predicted_char = labels[char_index]
print(predicted_char, confidence)
