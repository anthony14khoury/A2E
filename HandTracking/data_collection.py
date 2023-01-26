import os
import sys
import time
import inspect
import numpy

# Configurations to Install Leap
src_dir = os.path.dirname(inspect.getfile(inspect.currentframe()))
arch_dir = './x64'
sys.path.insert(0, os.path.abspath(os.path.join(src_dir, arch_dir)))
# print(os.path.abspath(os.path.join(src_dir, arch_dir)))

import Leap

class Params():
    def __init__(self):
        self.count = 0
        self.frame_store = []
        self.sequence = 0
        self.sequence_store = []
        self.EMPTY_HAND = [0] * 198
        self.LETTER = "z"
        self.FRAME_COUNT = 30
        self.SEQUENCE_COUNT = 10

def create_folders():
    DATA_PATH = os.path.join('DataCollection')
    actions = actions = numpy.array(['a', 'j', 'z', 'nothing'])

    for action in actions:
        try:
            os.makedirs(os.path.join(DATA_PATH, action))
        except:
            pass


def SampleListener(controller, params):
    
    count          = params.count
    frame_store    = params.frame_store
    sequence       = params.sequence
    sequence_store = params.sequence_store
    
    for i in range(params.SEQUENCE_COUNT):  # Looping through number of sequences
        
        for count in range(params.FRAME_COUNT): # Looping through number of frames
            
            frame = controller.frame()  # Get frame object
            
            if frame.is_valid:
                
                hands = frame.hands # Get hands object
                
                frame_store.append([])
                frame_store[count].extend([len(hands), len(frame.fingers)])
                
                leftHand, rightHand = [], []
                
                # Looping through hands
                for hand in hands:
                    if hand.is_left:
                        leftHand.extend(hand.palm_normal[0], hand.palm_normal[1], hand.palm_normal[2])
                        leftHand.extend(hand.direction[0], hand.direction[1], hand.direction[2])
                        leftHand.append(hand.direction.pitch * Leap.RAD_TO_DEG)
                        leftHand.append(hand.palm_normal.roll * Leap.RAD_TO_DEG)
                        leftHand.append(hand.direction.yaw * Leap.RAD_TO_DEG)
                        leftHand.extend(hand.arm.direction[0], hand.arm.direction[1], hand.arm.direction[2])
                        leftHand.extend(hand.arm.wrist_position[0], hand.arm.wrist_position[1], hand.arm.wrist_position[2])
                        leftHand.extend(hand.arm.elbow_position[0], hand.arm.elbow_position[1], hand.arm.elbow_position[2])

                        for finger in hand.fingers:
                            for boneIndex in range(0, 4):
                                bone = finger.bone(boneIndex) # Get bone object
                                leftHand.extend(bone.prev_joint[0], bone.prev_joint[1], bone.prev_joint[2])
                                leftHand.extend(bone.next_joint[0], bone.next_joint[1], bone.next_joint[2])
                                leftHand.extend(bone.direction[0], bone.direction[1], bone.direction[2])

                    else:
                        rightHand.extend(hand.palm_normal[0], hand.palm_normal[1], hand.palm_normal[2])
                        rightHand.extend(hand.direction[0], hand.direction[1], hand.direction[2])
                        rightHand.append(hand.direction.pitch * Leap.RAD_TO_DEG)
                        rightHand.append(hand.palm_normal.roll * Leap.RAD_TO_DEG)
                        rightHand.append(hand.direction.yaw * Leap.RAD_TO_DEG)
                        rightHand.extend(hand.arm.direction[0], hand.arm.direction[1], hand.arm.direction[2])
                        rightHand.extend(hand.arm.wrist_position[0], hand.arm.wrist_position[1], hand.arm.wrist_position[2])
                        rightHand.extend(hand.arm.elbow_position[0], hand.arm.elbow_position[1], hand.arm.elbow_position[2])


                        for finger in hand.fingers:
                            for boneIndex in range(0, 4):
                                bone = finger.bone(boneIndex)
                                rightHand.extend(bone.prev_joint[0], bone.prev_joint[1], bone.prev_joint[1])
                                rightHand.extend(bone.next_joint[0], bone.next_joint[1], bone.next_joint[1])
                                rightHand.extend(bone.direction[0], bone.direction[1], bone.direction[1])
                
                if len(leftHand) == 0:
                    frame_store[count] = frame_store[count] + params.EMPTY_HAND
                else:
                    frame_store[count] = frame_store[count] + leftHand
                if len(rightHand) == 0:
                    frame_store[count] = frame_store[count] + params.EMPTY_HAND
                else:
                    frame_store[count] = frame_store[count] + rightHand
                
                
            # Waiting in-between frames for 0.1 seconds
            time.sleep(.1)

        print("Done with gathering data for sequence: ", sequence)
        sequence_store.append(frame_store)
        sequence += 1
        frame_store = []
        count = 0

        print("Wait 3 seconds")
        time.sleep(3)
        print("Start again")
                
    print("Done with all sequences")
    
    print("Writing to numpy Files")
    target_folder = os.path.join(os.path.join('DataCollection'), params.LETTER)
    for i in range(0, params.SEQUENCE_COUNT):
        set_of_frames = numpy.array(sequence_store[i])
        numpy.save(target_folder + "/" + params.LETTER+str(i), set_of_frames)
    
    #Done gathering data
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
    create_folders()
    main()
