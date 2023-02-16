import Leap, sys, time, math, ctypes
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv


def convert_distortion_maps(image):
    distortion_length = image.distortion_width * image.distortion_height
    xmap = np.zeros(distortion_length / 2, dtype=np.float32)
    ymap = np.zeros(distortion_length / 2, dtype=np.float32)

    for i in range(0, distortion_length, 2):
        xmap[distortion_length / 2 - i / 2 - 1] = image.distortion[i] * image.width
        ymap[distortion_length / 2 - i / 2 - 1] = image.distortion[i + 1] * image.height

    xmap = np.reshape(xmap, (image.distortion_height, image.distortion_width / 2))
    ymap = np.reshape(ymap, (image.distortion_height, image.distortion_width / 2))

    resized_xmap = cv.resize(xmap,
                              (image.width, image.height),
                              0, 0,
                              cv.INTER_LINEAR)
    resized_ymap = cv.resize(ymap,
                              (image.width, image.height),
                              0, 0,
                              cv.INTER_LINEAR)

    coordinate_map, interpolation_coefficients = cv.convertMaps(resized_xmap,
                                                                 resized_ymap,
                                                                 cv.CV_32FC1,
                                                                 nninterpolation=False)

    return coordinate_map, interpolation_coefficients


def undistort(image, coordinate_map, coefficient_map, width, height):
    destination = np.empty((width, height), dtype=np.ubyte)
    i_address = int(image.data_pointer)
    ctype_array_def = ctypes.c_ubyte * image.height * image.width
    as_ctype_array = ctype_array_def.from_address(i_address)
    as_numpy_array = np.ctypeslib.as_array(as_ctype_array)
    img = np.reshape(as_numpy_array, (image.height, image.width))

    destination = cv.remap(img,
                            coordinate_map,
                            coefficient_map,
                            interpolation=cv.INTER_LINEAR)

    destination = cv.resize(destination,
                             (width, height),
                             0, 0,
                             cv.INTER_LINEAR)
    return destination

class SampleListener(Leap.Listener):

    def on_init(self, controller):
        print("Initialized")

    def on_connect(self, controller):
        print("Connected")

    def on_disconnect(self, controller):
        # Note: not dispatched when running in a debugger.
        print("Disconnected")

    def on_exit(self, controller):
        print("Exited")

    def on_frame(self, controller):
        # Get the most recent frame and report some basic information
        maps_initialized = False
        frame = controller.frame()
        image = frame.images[0]
        print(image)
        flag = True
        if image.is_valid:
            if not maps_initialized:
                right_coordinates, right_coefficients = convert_distortion_maps(frame.images[1])
                maps_initialized = True

            undistorted_right = undistort(image, right_coordinates, right_coefficients, 600, 600)

            # display images
            cv.imshow('Right Camera', undistorted_right)
            cv.waitKey(1)

def main():
    listener = SampleListener()
    controller = Leap.Controller()
    controller.set_policy(Leap.Controller.POLICY_BACKGROUND_FRAMES)
    controller.set_policy(Leap.Controller.POLICY_OPTIMIZE_HMD)
    controller.set_policy(Leap.Controller.POLICY_IMAGES)
    controller.add_listener(listener)
    print("Press Enter to quit...")
    try:
        sys.stdin.readline()
    except KeyboardInterrupt:
        pass

    finally:
        controller.remove_listener(listener)

if __name__ == "__main__":
    main()