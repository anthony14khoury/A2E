import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
from parameters import Params, mediapipe_detection, extract_keypoints

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

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


def collect_data(params, letter):
     
     # Set mediapipe model
     with mp_holistic.Holistic(model_complexity = 1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
          
          cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
          print("Camera is connected and Everything is Configured!")
          
          
          while cap.isOpened():
               
               for i in range(5):
                    print("Data Collection is starting in {} seconds!".format(5-i))
                    time.sleep(1.0)
                    ret, frame = cap.read()
               
               SEQUENCE_STORE = []
               for sequence in range(params.SEQUENCE_COUNT):
                    
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
                    
                    print("Done with Sequence: {}".format(sequence))
                    SEQUENCE_STORE.append(FRAME_STORE)
                    
                    print("Wait 2 seconds \n")
                    time.sleep(2.0)
                    
               
               print("Done with all Sequences for {}".format(letter))
               
               
               # Writing npy files
               target_folder = os.path.join(os.path.join(params.COLLECTION_FOLDER), letter)
               for i in range(params.SEQUENCE_COUNT):
                    set_of_frames = np.array(SEQUENCE_STORE[i])
                    np.save(target_folder + "/" + letter + str(i+40), set_of_frames)
                    
               
               print("\n Program is Finished \n")
               cap.release()
               cv2.destroyAllWindows()
               quit()
     
     
if __name__ == "__main__":
     
     params = Params()
     
     letter = 'e'
     
     # Create Folder
     try:
          os.makedirs(os.path.join(params.COLLECTION_FOLDER, letter))
     except:
          print("Folder already exists")
          pass
     
     collect_data(params, letter)