import os
import cv2
import sys
import time
import inspect
import numpy
from matplotlib import pyplot as plt
import Leap

# Configurations to Install Leap
src_dir = os.path.dirname(inspect.getfile(inspect.currentframe()))
arch_dir = './x64'
sys.path.insert(0, os.path.abspath(os.path.join(src_dir, arch_dir)))
print(os.path.abspath(os.path.join(src_dir, arch_dir)))

# Global Variables
count = 0
frame_store = []
sequence = 0
sequence_store = []
EMPTY_HAND = [[0,0,0],[0,0,0],0,0,0,[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]]
LETTER = "nothing"
FRAME_COUNT = 30
SEQUENCE_COUNT = 10

def create_folders():
    DATA_PATH = os.path.join('DataCollection')
    actions = actions = numpy.array(['a', 'nothing'])

    for action in actions:
        try:
            os.makedirs(os.path.join(DATA_PATH, action))
        except:
            pass


def SampleListener(controller):
    global count
    global frame_store
    global sequence
    global sequence_store
    while sequence < SEQUENCE_COUNT:    
        while count < FRAME_COUNT:
            frame = controller.frame()
            if frame.is_valid:
                frame_store.append([])
                hands = frame.hands
                frame_store[count].append(len(hands))
                frame_store[count].append(len(frame.fingers))
                leftHand = []
                rightHand = []
                for hand in hands:
                    if hand.is_left:
                        leftHand.append([hand.palm_normal[0], hand.palm_normal[1], hand.palm_normal[2]])
                        leftHand.append([hand.direction[0], hand.direction[1], hand.direction[2]])
                        leftHand.append(hand.direction.pitch * Leap.RAD_TO_DEG)
                        leftHand.append(hand.palm_normal.roll * Leap.RAD_TO_DEG)
                        leftHand.append(hand.direction.yaw * Leap.RAD_TO_DEG)
                        leftHand.append([hand.arm.direction[0], hand.arm.direction[1], hand.arm.direction[2]])
                        leftHand.append([hand.arm.wrist_position[0], hand.arm.wrist_position[1], hand.arm.wrist_position[2]])
                        leftHand.append([hand.arm.elbow_position[0], hand.arm.elbow_position[1], hand.arm.elbow_position[2]])
                        for finger in hand.fingers:
                            for boneIndex in range(0, 4):
                                bone = finger.bone(boneIndex)
                                leftHand.append([bone.prev_joint[0], bone.prev_joint[1], bone.prev_joint[2]])
                                leftHand.append([bone.next_joint[0], bone.next_joint[1], bone.next_joint[2]])
                                leftHand.append([bone.direction[0], bone.direction[1], bone.direction[2]])
                    else:
                        rightHand.append([hand.palm_normal[0], hand.palm_normal[1], hand.palm_normal[2]])
                        rightHand.append([hand.direction[0], hand.direction[1], hand.direction[2]])
                        rightHand.append(hand.direction.pitch * Leap.RAD_TO_DEG)
                        rightHand.append(hand.palm_normal.roll * Leap.RAD_TO_DEG)
                        rightHand.append(hand.direction.yaw * Leap.RAD_TO_DEG)
                        rightHand.append([hand.arm.direction[0], hand.arm.direction[1], hand.arm.direction[2]])
                        rightHand.append([hand.arm.wrist_position[0], hand.arm.wrist_position[1], hand.arm.wrist_position[2]])
                        rightHand.append([hand.arm.elbow_position[0], hand.arm.elbow_position[1], hand.arm.elbow_position[2]])
                        for finger in hand.fingers:
                            for boneIndex in range(0, 4):
                                bone = finger.bone(boneIndex)
                                rightHand.append([bone.prev_joint[0], bone.prev_joint[1], bone.prev_joint[2]])
                                rightHand.append([bone.next_joint[0], bone.next_joint[1], bone.next_joint[2]])
                                rightHand.append([bone.direction[0], bone.direction[1], bone.direction[2]])
                if len(leftHand) == 0:
                    frame_store[count] = frame_store[count] + EMPTY_HAND
                else:
                    frame_store[count] = frame_store[count] + leftHand
                if len(rightHand) == 0:
                    frame_store[count] = frame_store[count] + EMPTY_HAND
                else:
                    frame_store[count] = frame_store[count] + rightHand
                count += 1
                time.sleep(.1)

        print("Done with Gathering Data for sequence", sequence)
        sequence_store.append(frame_store)
        sequence += 1
        frame_store = []
        count = 0
        print("Wait 3 seconds")
        time.sleep(3)
        print("Start again")
                
    print("Done with all sequences")
    # Write to numpy files
    TargetFolder = os.path.join(os.path.join('DataCollection'), LETTER)
    for i in range(0,SEQUENCE_COUNT):
        set_of_frames = numpy.array(sequence_store[i])
        numpy.save(TargetFolder+"/"+LETTER+str(i), set_of_frames)
    quit()


def main():
    # Create a controller
    controller = Leap.Controller()

    while not controller.is_connected:
        pass
    print ("Connected")
    print("Start!")
    # Keep this process running until Enter is pressed
    print ("Press Enter to quit...")
    SampleListener(controller)

if __name__ == "__main__":
    create_folders()
    main()
