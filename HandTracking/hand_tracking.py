import os
import cv2
import sys
import time
import inspect
import numpy
import numpy as np
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
EMPTY_HAND = [[0,0,0],[0,0,0],0,0,0,[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]]
LETTER = 'a'
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
            else:
                print("Done with Gathering Data for sequence", sequence)
                sequence_store.append(frame_store)
                sequence += 1
                if sequence < 10:
                    frame_store = []
                    count = 0
                    print("Wait 3 seconds")
                    time.sleep(3)
                    print("Start again")
                else:
                    print("Done with all sequences")
                    # Write to numpy files
                    TargetFolder = os.path.join(os.path.join('DataCollection'), LETTER)
                    for i in range(0,10):
                        set_of_frames = numpy.asarray(sequence_store[i])
                        numpy.save(TargetFolder+"/"+LETTER+str(i), set_of_frames)
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
