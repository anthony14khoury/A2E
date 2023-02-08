import os
import time
import numpy as np
from Clean import hand_tracking, Params
import inspect, sys
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import SGD
import keras
from keras.models import load_model
from keras.utils import CustomObjectScope

# Configurations to Install Leap
src_dir = os.path.dirname(inspect.getfile(inspect.currentframe()))
arch_dir = './x64'
sys.path.insert(0, os.path.abspath(os.path.join(src_dir, arch_dir)))

import Leap

model = load_model('g_l_nothing.h5')# #getting the labels form data directory
# labels = sorted(os.listdir("DataCollection"))
labels = ['g', 'l', 'nothing']

def SampleListener(controller, params):
    
     while True:
        
          params.frame_store = []
        
          for count in range(30):  # Looping through number of sequences
            
               frame = controller.frame()  # Get frame object
               if frame.is_valid:
                    
                    params.frame_store = hand_tracking(frame, params.frame_store, count, params, False)
               
               time.sleep(0.1)

          # predict
          # input = np.array(frame_store)
          prediction = model.predict(np.expand_dims(params.frame_store,axis=0))
          char_index = np.argmax(prediction)
          confidence = round(prediction[0,char_index]*100, 1)
          predicted_char = labels[char_index]
          print(predicted_char, confidence)


def main():
    
     # Class Instantiation
     params = Params()
     
     # Create a controller
     controller = Leap.Controller()

     print("Waiting for controller to connect")
     while not controller.is_connected:
          pass
     
     print ("Controller is connected")
     time.sleep(2)

     # Keep this process running until Enter is pressed
     SampleListener(controller, params)

if __name__ == "__main__":
    main()