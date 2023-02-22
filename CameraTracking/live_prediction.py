from keras.models import load_model
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
from parameters import Params, mediapipe_detection, extract_keypoints
import socket

from word_detection import add_spaces

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


# constants for prediction script
getReady = "Prepare to Sign!"
go = "Go!"
font = cv2.FONT_HERSHEY_SIMPLEX
color = (255,0,255)
fontScale = 2
thickness = 4

# socket settings
HOST = "10.136.49.55" # The server's hostname or IP address
PORT =4000 # The port used by the server

def generic_mediapipe_detectiqon(image, model):
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
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten()
    else:
        lh = np.zeros(21*3)

    if results.right_hand_landmarks:
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten()
    else:
        rh = np.zeros(21*3)

    return np.concatenate([lh, rh])

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

def prediction(params, model, letters):

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        # connect to socket
        #s.connect((HOST, PORT))

        # Set mediapipe model
        with mp_holistic.Holistic(model_complexity = 1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

            cap = cv2.VideoCapture(0, cv2.CAP_ANY)

            # Wait until camera is connected
            while not cap.isOpened():
                pass

            print("\nCamera is connected and Everything is Configured!")

            print("Beginning Predictions:\n")
            while cap.isOpened():
                FRAME_STORE = []
                for frame_num in range(params.FRAME_COUNT):
                    start = time.time()
		            
                    # Read Feed         

                    ret, frame = cap.read()
                    start = time.time()
                    # Made detections
                    image, results = mediapipe_detection(frame, holistic)
                    #draw_styled_landmarks(image, results)
                    keypoints = extract_keypoints(results)
                    FRAME_STORE.append(keypoints)

                    #if frame_num < 10:
                     #   image = cv2.putText(image, go, (int(len(image[0])/2)-10, int(len(image)/2)), font, fontScale, color, thickness, cv2.LINE_AA)
                     #   cv2.imshow('OpenCV Feed', image)
                        # Breaking gracefully
                      #  if cv2.waitKey(5) & 0xFF == ord('q'):
                       #     cap.release()
                        #    cv2.destroyAllWindows()
                         #   quit()

                    # Just Show to screen
                    #cv2.imshow('OpenCV Feed', image)

                    # Breaking gracefully
                    #if cv2.waitKey(5) & 0xFF == ord('q'):
                     #   cap.release()
                      #  cv2.destroyAllWindows()
                       # quit()

                    print(time.time()-start)

                prediction = model.predict(np.expand_dims(FRAME_STORE, axis=0))
                char_index = np.argmax(prediction)
                confidence = round(prediction[0,char_index]*100, 1)
                predicted_char = letters[char_index]
                print(predicted_char, confidence)
               
                timeout = time.time() + 2
                while True:
                        
                    if time.time() > timeout:
                        break
    
                    # Read Feed
                    ret, frame = cap.read()
                    
                    # Made detections
                    image, results = mediapipe_detection(frame, holistic)
                    draw_styled_landmarks(image, results)
                    keypoints = extract_keypoints(results)
                    FRAME_STORE.append(keypoints)
                    
                    # Show to screen
                    image = cv2.putText(image, getReady, (int(len(image[0])/2)-200, int(len(image)/2)), font, fontScale, color, thickness, cv2.LINE_AA)
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
                # s.send(predicted_char)
               
                if len(predicted_char) > 1:
                  #ipdb.set_trace()
                  curr_sen = np.concatenate((curr_sen,temp))
                  curr_letters = ""
                  curr_sen = np.append(curr_sen,predicted_char)
                  temp = []
                else:
                  curr_letters += predicted_char
                answer,temp = add_spaces(curr_letters, curr_sen)         #Print out most likely placement of spaces, add dashes if none found
                print(answer)
          
                # print("New Collection:")
                print("Wait 2 seconds \n")
                time.sleep(2.0)
          

                print("Wait 2 seconds \n")
               
                #image = cv2.putText(image, getReady, (int(len(image[0])/2)-200, int(len(image)/2)), font, fontScale, color, thickness, cv2.LINE_AA)

                #cv2.imshow('OpenCV Feed', image)
                # Breaking gracefully
                #if cv2.waitKey(5) & 0xFF == ord('q'):
                 #   cap.release()
                  #  cv2.destroyAllWindows()
                   # quit()

                time.sleep(2.0)

if __name__ == "__main__":
     
    # Load in the ML Model
    model = load_model('model3.h5')

    # Class Instantiation
    params = Params()
    
    # Keep this process running until Enter is pressed
    prediction(params, model, params.LETTERS)
