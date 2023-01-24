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


# Global Variables
count = 0
frame_store = []
sequence = 0
sequence_store = []

def create_folders():
     DATA_PATH = os.path.join('DataCollection')
     actions = actions = np.array(['a', 'nothing'])
     
     for action in actions: 
          try: 
               os.makedirs(os.path.join(DATA_PATH, action))
          except:
               pass


class SampleListener(Leap.Listener):
     def on_connect(self, controller):
          print ("Connected")

     def on_frame(self, controller):

          global count
          global frame_store
          global sequence
          global sequence_store
          frame = controller.frame()
          
          if frame.is_valid:
               if count < 30:                    
                    # Variables to grab
                    
                    count += 1
                    
               else:
                    print("Done with Gathering Data for sequence", sequence)
                    sequence += 1
                    if sequence < 10:
                         sequence_store.append(frame_store)
                         frame_store = []
                         count = 0
                         print("Wait 3 seconds")
                         sleep(3)
                         print("Start again")
                    else:
                         print("Done with all sequences")
                         #print data to files
                         
                         quit()
               

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
     create_folders()
     main()
