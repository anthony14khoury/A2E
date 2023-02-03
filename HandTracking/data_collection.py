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
    
    for i in range(params.SEQUENCE_COUNT):  # Looping through number of sequences
        
        for count in range(params.FRAME_COUNT): # Looping through number of frames
            
            frame = controller.frame()  # Get frame object
            
            if frame.is_valid:
                
                params.frame_store = hand_tracking(frame, params.frame_store, count, params)
                
            time.sleep(0.1)

        print("Done with gathering data for sequence: ", params.sequence)
        params.sequence_store.append(params.frame_store)
        params.sequence += 1
        params.frame_store = []


        print("Wait 3 seconds")
        time.sleep(3)
        print("Start again")

   
    print("Done with all sequences")
    
    
    
    print("Writing to numpy Files")
    target_folder = os.path.join(os.path.join('DataCollection'), params.LETTER)
    for i in range(0, params.SEQUENCE_COUNT):
        set_of_frames = numpy.array(params.sequence_store[i])
        numpy.save(target_folder + "/" + params.LETTER+str(i), set_of_frames)


    quit()


def main():
    
    # Class Instantiation
    params = Params()
    
    # Create a controller
    controller = Leap.Controller()
    
    print("Waiting for controller to connect")
    while not controller.is_connected:
        pass
    time.sleep(5)
    
    print ("Controller is connected")
    print("Start!")
    
    SampleListener(controller, params)


if __name__ == "__main__":
    
    actions = ['a', 'j', 'z', 'nothing']
    folder = 'DataCollection'
    
    create_folders(folder, actions)
    main()
