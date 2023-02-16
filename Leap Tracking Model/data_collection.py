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
    
    # Looping through signs
    for sign in params.SIGNS:
    
        # Looping through number of sequences (20)
        for i in range(params.SEQUENCE_COUNT):
            
            print("Start Data Collection for letter: {} | Sequence: {}".format(sign, i))
            
            # Looping through number of frames (30)
            for count in range(params.FRAME_COUNT):
                
                # Accessing Frame Object from Leap API
                frame = controller.frame()
                
                # Checking if the Frame is Valid
                if frame.is_valid:
                    
                    # Function Retuning all of the Tracked Values (438)
                    params.FRAME_STORE = hand_tracking(frame, params.FRAME_STORE, count, params, True)
                    
                    # Allows for a consistent capture of frames
                    time.sleep(0.1)
                
                else:
                    print("Error: frame is not valid")
                    quit()
                    
            print("Done \n")
            params.SEQUENCE_STORE.append(params.FRAME_STORE)
	    params.FRAME_STORE = []

            print("Wait 1 seconds \n")
            time.sleep(0.0)

        print("Done with All Sequences for {}".format(sign))
        
        print("Writing to numpy Files")
        target_folder = os.path.join(os.path.join('DataCollection'), sign)
        for i in range(params.SEQUENCE_COUNT):
            set_of_frames = numpy.array(params.SEQUENCE_STORE[i])
            numpy.save(target_folder + "/" + sign + str(i), set_of_frames)
            
            
        # Clearing Variables for Next Letter of Data Collection
        params.SEQUENCE_STORE = []

    print("\n Program is Finished \n")
    quit()
    


if __name__ == "__main__":
    print("Program is Running:")
    
    signs = ['c']
    folder = 'DataCollection'
    
    print("\t Create Folders for Signs")
    create_folders(folder, signs)
    
    # Class Instantiation
    params = Params()
    params.SIGNS = signs
    
    # # Create a controller
    controller = Leap.Controller()
    
    print("\t Waiting for controller to connect")
    while not controller.is_connected:
        pass
    print ("\t Controller is connected")
    
    
    print("\t Double Checking the Connection")
    time.sleep(3.0)
    
    print("\t Done and Starting Data Collection \n")
    
    SampleListener(controller, params)
