import os, sys, inspect
import numpy

# Configurations to Install Leap
src_dir = os.path.dirname(inspect.getfile(inspect.currentframe()))
arch_dir = './x64'
sys.path.insert(0, os.path.abspath(os.path.join(src_dir, arch_dir)))

import Leap

class Params():
     def __init__(self):
          # self.count = 0
          self.FRAME_STORE = []
          # self.sequence = 0
          self.SEQUENCE_STORE = []
          self.EMPTY_HAND = [0] * 198
          # self.LETTER = ""
          self.FRAME_COUNT = 30
          self.SEQUENCE_COUNT = 20
          self.SIGNS = []


def create_folders(folder, actions):
     DATA_PATH = os.path.join(folder)
     actions = actions = numpy.array(actions)

     for action in actions:
          try:
               os.makedirs(os.path.join(DATA_PATH, action))
          except:
               pass
          
def hand_tracking(frame, frame_store, count, params, detail):
     
     # Accessing Hand Object from Leap API
     hands = frame.hands
     
     frame_store.append([])
     frame_store[count].extend([len(hands), len(frame.fingers)])
     
     leftHand, rightHand = [], []
     
     # Looping through hands
     for hand in hands:
          if hand.is_left:
               leftHand.extend([hand.palm_normal[0], hand.palm_normal[1], hand.palm_normal[2]])
               leftHand.extend([hand.direction[0], hand.direction[1], hand.direction[2]])
               leftHand.append(hand.direction.pitch * Leap.RAD_TO_DEG)
               leftHand.append(hand.palm_normal.roll * Leap.RAD_TO_DEG)
               leftHand.append(hand.direction.yaw * Leap.RAD_TO_DEG)
               leftHand.extend([hand.arm.direction[0], hand.arm.direction[1], hand.arm.direction[2]])
               leftHand.extend([hand.arm.wrist_position[0], hand.arm.wrist_position[1], hand.arm.wrist_position[2]])
               leftHand.extend([hand.arm.elbow_position[0], hand.arm.elbow_position[1], hand.arm.elbow_position[2]])

               for finger in hand.fingers:
                    for boneIndex in range(0, 4):
                         bone = finger.bone(boneIndex) # Get bone object
                         leftHand.extend([bone.prev_joint[0], bone.prev_joint[1], bone.prev_joint[2]])
                         leftHand.extend([bone.next_joint[0], bone.next_joint[1], bone.next_joint[2]])
                         leftHand.extend([bone.direction[0], bone.direction[1], bone.direction[2]])

          else:
               rightHand.extend([hand.palm_normal[0], hand.palm_normal[1], hand.palm_normal[2]])
               rightHand.extend([hand.direction[0], hand.direction[1], hand.direction[2]])
               rightHand.append(hand.direction.pitch * Leap.RAD_TO_DEG)
               rightHand.append(hand.palm_normal.roll * Leap.RAD_TO_DEG)
               rightHand.append(hand.direction.yaw * Leap.RAD_TO_DEG)
               rightHand.extend([hand.arm.direction[0], hand.arm.direction[1], hand.arm.direction[2]])
               rightHand.extend([hand.arm.wrist_position[0], hand.arm.wrist_position[1], hand.arm.wrist_position[2]])
               rightHand.extend([hand.arm.elbow_position[0], hand.arm.elbow_position[1], hand.arm.elbow_position[2]])


               for finger in hand.fingers:
                    for boneIndex in range(0, 4):
                         bone = finger.bone(boneIndex)
                         rightHand.extend([bone.prev_joint[0], bone.prev_joint[1], bone.prev_joint[1]])
                         rightHand.extend([bone.next_joint[0], bone.next_joint[1], bone.next_joint[1]])
                         rightHand.extend([bone.direction[0], bone.direction[1], bone.direction[1]])
          
     if len(leftHand) == 0:
          frame_store[count] = frame_store[count] + params.EMPTY_HAND
     else:
          frame_store[count] = frame_store[count] + leftHand
     
     if len(rightHand) == 0:
          frame_store[count] = frame_store[count] + params.EMPTY_HAND
     else:
          frame_store[count] = frame_store[count] + rightHand
     
     if detail == True:
          
          print("Number of Hands Present: ", len(hands))
          for hand in hands:
               if hand.is_left:
                    # print("Left Hand Present")
                    print("Number of finders detected in left hand: ", len(hand.fingers))
                         
               elif hand.is_right:
                    # print("Right Hand Present")
                    print("Number of finders detected in right hand: ", len(hand.fingers))
     
     return frame_store