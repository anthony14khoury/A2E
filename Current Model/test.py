from time import sleep
from leap_data_helper import *
import numpy as np
import sys
import ctypes
#import tensorflow as tf
import os
import Leap as Leap

coordinate_map = []
coordniate_coefficients = []
maps_initialized = False
#model = tf.keras.models.load_model(os.path.abspath(os.getcwd()) + "/model.h5")
#data_dir = 'Image_Directory'
##getting the labels form data directory
#labels = sorted(os.listdir(data_dir))



def convert_distortion_maps(image):

    distortion_length = image.distortion_width * image.distortion_height
    xmap = np.zeros(distortion_length/2, dtype=np.float32)
    ymap = np.zeros(distortion_length/2, dtype=np.float32)
    
    for i in range(0, distortion_length, 2):
        xmap[distortion_length/2 - i/2 - 1] = image.distortion[i] * image.width
        ymap[distortion_length/2 - i/2 - 1] = image.distortion[i + 1] * image.height

    xmap = np.reshape(xmap, (image.distortion_height, image.distortion_width/2))
    ymap = np.reshape(ymap, (image.distortion_height, image.distortion_width/2))

    #resize the distortion map to equal desired destination image size
    resized_xmap = cv2.resize(xmap,
                              (image.width, image.height),
                              0, 0,
                              cv2.INTER_LINEAR)
    resized_ymap = cv2.resize(ymap,
                              (image.width, image.height),
                              0, 0,
                              cv2.INTER_LINEAR)

    #Use faster fixed point maps
    coordinate_map, interpolation_coefficients = cv2.convertMaps(resized_xmap,
                                                                 resized_ymap,
                                                                 cv2.CV_32FC1,
                                                                 nninterpolation = False)

    return coordinate_map, interpolation_coefficients

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
    destination = np.empty((width, height), dtype = np.ubyte)

    #wrap image data in numpy array
    i_address = int(image.data_pointer)
    ctype_array_def = ctypes.c_ubyte * image.height * image.width
    # as ctypes array
    as_ctype_array = ctype_array_def.from_address(i_address)
    # as numpy array
    as_numpy_array = np.ctypeslib.as_array(as_ctype_array)
    img = np.reshape(as_numpy_array, (image.height, image.width))

    #remap image to destination
    destination = cv2.remap(img,
                            coordinate_map,
                            coefficient_map,
                            interpolation = cv2.INTER_LINEAR)

    #resize output to desired destination size
    destination = cv2.resize(destination,
                             (width, height),
                             0, 0,
                             cv2.INTER_LINEAR)
    return destination

def SampleListener(controller):
    while(True):
        global maps_initialized
        global coordinate_map
        global coordniate_coefficients

        while(not controller.is_connected):
            pass

        frame = controller.frame()
        print(controller)
        if frame.is_valid:
            image = frame.images[0]
            if not maps_initialized:
                random = image.distortion_width
                random2 = image.height
                print(random)
                distortion_length = image.distortion_width * image.distortion_height
                xmap = np.zeros(distortion_length/2, dtype=np.float32)
                ymap = np.zeros(distortion_length/2, dtype=np.float32)
                
                for i in range(0, distortion_length, 2):
                    xmap[distortion_length/2 - i/2 - 1] = image.distortion[i] * image.width
                    ymap[distortion_length/2 - i/2 - 1] = image.distortion[i + 1] * image.height

                xmap = np.reshape(xmap, (image.distortion_height, image.distortion_width/2))
                ymap = np.reshape(ymap, (image.distortion_height, image.distortion_width/2))

                #resize the distortion map to equal desired destination image size
                resized_xmap = cv2.resize(xmap,
                                        (image.width, image.height),
                                        0, 0,
                                        cv2.INTER_LINEAR)
                resized_ymap = cv2.resize(ymap,
                                        (image.width, image.height),
                                        0, 0,
                                        cv2.INTER_LINEAR)

                #Use faster fixed point maps
                coordinate_map, coordniate_coefficients = cv2.convertMaps(resized_xmap,
                                                                            resized_ymap,
                                                                            cv2.CV_32FC1,
                                                                            nninterpolation = False)
                maps_initialized = True
            #format received image
            img = undistort(image, coordinate_map, coordniate_coefficients, 400, 400)
            img[img<60] = 0
            img = hand_cropping(img)
            img = np.expand_dims(img, 2)
            img = resize_img(img, 32)
            img = normalize_data(img)
            #make predication about the current frame
            # prediction = model.predict(img.reshape(1,50,50,3))
            # char_index = np.argmax(prediction)
            # confidence = round(prediction[0,char_index]*100, 1)
            # predicted_char = labels[char_index]
            # print(predicted_char, confidence)

def main():
    controller = Leap.Controller()
    controller.set_policy(Leap.Controller.POLICY_IMAGES)
 # Keep this process running until Enter is pressed
    print ("Press Enter to quit...")
    try:
        SampleListener(controller)
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
