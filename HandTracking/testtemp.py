import os
import sys
import time
import inspect
import numpy
from Clean import Params, create_folders, hand_tracking

# Configurations to Install Leap
src_dir = os.path.dirname(inspect.getfile(inspect.currentframe()))
arch_dir = './x64'
sys.path.insert(0, os.path.abspath(os.path.join(src_dir, arch_dir)))

import Leap

def SampleListener(controller, params):
    while True:
          
          frame = controller.frame()  # Get frame object
            
          if frame.is_valid:
               
               params.frame_store = hand_tracking(frame, params.frame_store, 0, params, True)
               # print(len(frame.hands))
               
               time.sleep(1)


def main():
     
     # Class Instantiation
     params = Params()
     
     # Create a controller
     controller = Leap.Controller()
     
     print("Waiting for controller to connect")
     while not controller.is_connected:
          pass
     print ("Controller is connected")
     
     print("Waiting 3 seconds to ensure connection")
     time.sleep(3)
     
     print("Start!")
     
     SampleListener(controller, params)


if __name__ == "__main__":
    main()