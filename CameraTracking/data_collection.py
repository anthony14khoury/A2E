from parameters import Params, extract_hand_keypoints
import mediapipe as mp
import numpy as np
import time
import cv2
import os

# Mediapipe Variables
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Parameter Class
params = Params()

# Collection Variables
letter = 'a'
collection_folder = 'HandsCollection'

# Collection Types: "video" or "static"
type = "static"


# Constants for Prediction Script
# getReady = "Prepare to Sign!"
# go = "Go!"
# font = cv2.FONT_HERSHEY_SIMPLEX
# color = (255,0,255)
# fontScale = 2
# thickness = 4


# Create Folder
try:
     os.makedirs(os.path.join(collection_folder, letter))
except:
     print("Folder already exists")
     pass


def draw_styled_landmarks(image, results):
     if results.multi_hand_landmarks:
          for hand_landmarks in results.multi_hand_landmarks:
               mp_drawing.draw_landmarks(
               image, 
               hand_landmarks,
               mp_hands.HAND_CONNECTIONS,
               mp_drawing_styles.get_default_hand_landmarks_style(),
               mp_drawing_styles.get_default_hand_connections_style())
     return image
          

if type == "video":

     # Use CV2 Functionality to create a Video Stream
     # cap = cv2.VideoCapture(0, cv2.CAP_ANY)
     cap = cv2.VideoCapture(0)
     while not cap.isOpened():
          pass
     print("Camera is connected")

     with mp_hands.Hands(model_complexity=0, min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=2) as hands:     
          
          while cap.isOpened():
               
               for i in range(5):
                    print("Data Collection is starting in {} seconds!".format(5-i))
                    time.sleep(1.0)
                    ret, frame = cap.read()
               
               SEQUENCE_STORE = []
               for sequence in range(params.SEQUENCE_COUNT):
                    
                    FRAME_STORE = []
                    for frame_num in range(params.FRAME_COUNT):
                                                  
                         # Capture a frame
                         success, image = cap.read()
                         
                         # Error Checking
                         if not success:
                              print("Ignoring Empty Camera Frame")
                              continue
                         
                         # Made detections
                         image.flags.writeable = False
                         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                         results = hands.process(image)
                         
                         image.flags.writeable = True
                         image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                         image = draw_styled_landmarks(image, results)
                         FRAME_STORE.append(extract_hand_keypoints(results))
                         
                         # Display Image
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
               
               
               """ Writing Files """
               target_folder = os.path.join(os.path.join(collection_folder), letter)
               for i in range(params.SEQUENCE_COUNT):
                    set_of_frames = np.array(SEQUENCE_STORE[i])
                    np.save(target_folder + "/" + letter + str(i+0), set_of_frames)
                    
               
               print("\n Program is Finished \n")
               cap.release()
               cv2.destroyAllWindows()
               quit()
               
               
if type == "static":
     
     with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5) as hands:
     
          # Read in Image
          image_loc = "D:/Code/A2E/asl_dataset/asl_alphabet_train/asl_alphabet_train/A/A1.jpg"
          image = cv2.imread(image_loc)
          
          # Made detections
          image.flags.writeable = False
          image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
          results = hands.process(image)
          
          print('Handedness:', results.multi_handedness)
          
          image.flags.writeable = True
          image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
          image = draw_styled_landmarks(image, results)
          results = extract_hand_keypoints(results)
          # FRAME_STORE.append(extract_hand_keypoints(results))
          
          
          # Display Image
          
          # cv2.imshow('OpenCV Feed', image)
          # cv2.imwrite('./A1.png', image)
          print(results)
          
          
               
               
               
               
               
               
               
               
