import tensorflow as tf
import os
import Leap as Leap
import time
import numpy as np

model = tf.keras.models.load_model("action.h5")
#getting the labels form data directory
labels = sorted(os.listdir("DataCollection"))

def SampleListener(controller, params):
    while True:
        frame_store = []
        for count in range(30):  # Looping through number of sequences
            frame = controller.frame()  # Get frame object
            if frame.is_valid:
                frame_store.append([])
                hands = frame.hands     # Get hands object
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

        # predict
        prediction = model.predict(frame_store)
        char_index = np.argmax(prediction)
        confidence = round(prediction[0,char_index]*100, 1)
        predicted_char = labels[char_index]
        print(predicted_char, confidence)


def main():
    # Create a controller
    controller = Leap.Controller()

    print("Waiting for controller to connect")
    while not controller.is_connected:
        pass

    print ("Controller is connected")

    # Keep this process running until Enter is pressed
    SampleListener(controller)

if __name__ == "__main__":
    main()