import os
import cv2
import sys
import time
import inspect
import numpy as np
from matplotlib import pyplot as plt

# Configurations to Install Leap
src_dir = os.path.dirname(inspect.getfile(inspect.currentframe()))
arch_dir = './x64'
sys.path.insert(0, os.path.abspath(os.path.join(src_dir, arch_dir)))
print(os.path.abspath(os.path.join(src_dir, arch_dir)))

import Leap


class SampleListener(Leap.Listener):
     def on_connect(self, controller):
          print ("Connected")

     def on_frame(self, controller):
          global maps_initialized
          global coordinate_map
          global coordniate_coefficients

          frame = controller.frame()
          # if frame.is_valid:
          print("Frame ID: ", frame.id)

     def on_exit(self, controller):
          print ("Exited")


def main():
     # Create a sample listener and controller
     listener = SampleListener()
     controller = Leap.Controller()

     # Have the sample listener receive events from the controller
     controller.add_listener(listener)

     # Keep this process running until Enter is pressed
     print("Press Enter to quit...")
     try:
          sys.stdin.readline()
     
     except KeyboardInterrupt:
          pass
     
     finally:
          # Remove the sample listener when done
          controller.remove_listener(listener)
     


if __name__ == "__main__":
    main()
