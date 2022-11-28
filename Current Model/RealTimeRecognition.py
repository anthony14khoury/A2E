from leap_data_helper import *
import numpy as np
import sys
import tensorflow as tf
import os
import Leap as Leap

model = tf.keras.models.load_model(os.path.abspath(os.getcwd()) + "/model.h5")
data_dir = 'Image_Directory'
#getting the labels form data directory
labels = sorted(os.listdir(data_dir))

def hand_cropping(img):

    img = img[200:400, 200:400]

    dist_x = np.sum(img,0)
    dist_y = np.sum(img,1)
    span_x = np.where(dist_x>500)
    span_x_start = np.min(span_x)
    span_x_end = np.max(span_x)
    span_y = np.where(dist_y>50)
    span_y_start = np.min(span_y)
    span_y_end = np.max(span_y)

    if len(span_y[0])/len(span_x[0]) > 2:
        span_y_end = int(span_y_start + len(span_x[0])*1.8)
    return img[span_y_start:span_y_end+1,span_x_start:span_x_end+1]

def undistort(image, coordinate_map, coefficient_map, width, height):
    #remap image to destination
    destination = cv2.remap(image,
                            coordinate_map,
                            coefficient_map,
                            interpolation = cv2.INTER_LINEAR)

    #resize output to desired destination size
    destination = cv2.resize(destination,
                             (width, height),
                             0, 0,
                             cv2.INTER_LINEAR)
    return destination

class SampleListener(Leap.Listener):

    def on_connect(self, controller):
        print ("Connected")

    def on_frame(self, controller):
        print ("Frame available")
        frame = controller.frame()

        if frame.is_valid:
            #format received image
            img = frame.images[0]
            distortion_buffer = img.Distortion()
            left_coefficients = distortion_buffer['left_coefficients']
            img = undistort(img, distortion_buffer, left_coefficients, 640, 640)
            img[img<60] = 0
            img = hand_cropping(img)
            img = np.expand_dims(img, 2)
            img = resize_img(img, 32)
            img = normalize_data(img)

            #make predication about the current frame
            prediction = model.predict(img.reshape(1,50,50,3))
            char_index = np.argmax(prediction)
            confidence = round(prediction[0,char_index]*100, 1)
            predicted_char = labels[char_index]
            print(predicted_char, confidence)


    def on_exit(self, controller):
        print ("Exited")

listener = SampleListener()
controller = Leap.Controller()

controller.set_policy_flags(Leap.Controller.POLICY_IMAGES)

# Sample listener receive events from the controller
controller.add_listener(listener)

# Keep this process running until Enter is pressed
print ("Press Enter to quit...")
try:
    sys.stdin.readline()
except KeyboardInterrupt:
    pass
finally:
    # Remove the sample listener when done
    controller.remove_listener(listener)