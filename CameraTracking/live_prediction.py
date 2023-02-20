from keras.models import load_model
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
from parameters import Params

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

def generic_mediapipe_detection(image, model):
     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Color conversion from BGR -> RGB
     image.flags.writeable = False                  # Image is no longer writable
     image.flags.writeable = True                   # Image is now writable
     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # Color conversion from RGB -> BGR
     return image

def mediapipe_detection(image, model):
     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Color conversion from BGR -> RGB
     image.flags.writeable = False                  # Image is no longer writable
     results = model.process(image)                 # Make Prediction
     image.flags.writeable = True                   # Image is now writable
     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # Color conversion from RGB -> BGR
     return image, results

def extract_keypoints(results):
     
     if results.left_hand_landmarks: 
          left_hand = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten()
     else: 
          left_hand = np.zeros(21*3)
          
     if results.right_hand_landmarks: 
          rignt_hand = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten()
     else: 
          rignt_hand = np.zeros(21*3)
               
     # if results.pose_landmarks: 
     #      pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten()
     #      pose = pose[(11*4)+1 : (23*4)+1] # Only getting the relevant data from pose
     # else: 
     #      pose = np.zeros(48)
          
     # return np.concatenate([left_hand, rignt_hand, pose])
     return np.concatenate([left_hand, rignt_hand])


def draw_styled_landmarks(image, results):
     
     # Draw Left Hand Connections
     mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                              mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                              mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                              ) 
     # Draw Right Hand Connections  
     mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                              mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                              mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                              )
     # Draw Pose Connections
     # mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
     #                          mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
     #                          mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
     #                          )

def prediction(params, model, letters):
                     
     # Set mediapipe model
     with mp_holistic.Holistic(model_complexity = 1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
          
          # cap = cv2.VideoCapture(0, cv2.CAP_ANY)
          cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
          print("\nCamera is connected and Everything is Configured!")
          print("Beginning Predictions:\n")
          
          while cap.isOpened():
               
               FRAME_STORE = []
               for frame_num in range(params.FRAME_COUNT):
                                        
                    # Read Feed
                    ret, frame = cap.read()
                    
                    # Made detections
                    image, results = mediapipe_detection(frame, holistic)
                    draw_styled_landmarks(image, results)
                    keypoints = extract_keypoints(results)
                    FRAME_STORE.append(keypoints)
                    
                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)
                    
                    # Breaking gracefully
                    if cv2.waitKey(5) & 0xFF == ord('q'):
                         cap.release()
                         cv2.destroyAllWindows()
                         quit()
                    

               prediction = model.predict(np.expand_dims(FRAME_STORE, axis=0))
               char_index = np.argmax(prediction)
               confidence = round(prediction[0,char_index]*100, 1)
               predicted_char = letters[char_index]
               # print(prediction)
               print(predicted_char, confidence)
          
               # print("New Collection:")
               print("Wait 2 seconds \n")
               time.sleep(2.0)


if __name__ == "__main__":
     
     # Load in the ML Model
     model = load_model('test_model1.h5')
    
     # Class Instantiation
     params = Params()

     # Keep this process running until Enter is pressed
     prediction(params, model, params.LETTERS)