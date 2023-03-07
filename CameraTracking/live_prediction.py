from keras.models import load_model
import cv2
import numpy as np
import time
import mediapipe as mp
from parameters import Params, mediapipe_detection, extract_hand_keypoints
import socket
# from word_detection import add_spaces

# Parameter Class
params = Params()

# ML Model
model = load_model('./A2E/CameraTracking/Models/128_model_tanh_6.h5')
letters = params.LETTERS

# Mediapipe Modules
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Constants for Prediction Script
getReady = "Prepare to Sign!"
go = "Go!"
font = cv2.FONT_HERSHEY_SIMPLEX
color = (255,0,255)
fontScale = 1
thickness = 2

# Socket Settings
HOST = "10.136.49.55" # The server's hostname or IP address
PORT = 4000 # The port used by the server


def draw_styled_landmarks(image, results):
    
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
            
    return image


# Use CV2 Functionality to create a Video Stream
# cap = cv2.VideoCapture(0, cv2.CAP_ANY)
cap = cv2.VideoCapture(0)
while not cap.isOpened():
    pass
print("Camera is connected")

with mp_hands.Hands(model_complexity=0, min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=2) as hands:

    # Stay open while the camera is activated
    while cap.isOpened():
        
        FRAME_STORE = []
        
        # Loop through all of the frames
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
            # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image = draw_styled_landmarks(image, results)
            FRAME_STORE.append(extract_hand_keypoints(results))
            
            # Display Image
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
        print(predicted_char, confidence)
        
        # if len(predicted_char) > 1:
        #     #ipdb.set_trace()
        #     curr_sen = np.concatenate((curr_sen,temp))
        #     curr_letters = ""
        #     curr_sen = np.append(curr_sen,predicted_char)
        #     temp = []
        # else:
        #     curr_letters += predicted_char
        #     answer,temp = add_spaces(curr_letters, curr_sen)  #Print out most likely placement of spaces, add dashes if none found
        
        # print(answer)
        
        
        """ Continuous Camera Share """
        timeout = time.time() + 2
        while True:
                    
            if time.time() > timeout:
                break

            # Read Feed
            ret, frame = cap.read()
                
            # Show to screen
            to_screen = "{}: {}%".format(predicted_char, confidence)
            image = cv2.putText(image, to_screen, (int(len(image[0])/2)-200, int(len(image)/2)), font, fontScale, color, thickness, cv2.LINE_AA)
            cv2.imshow('OpenCV Feed', image)
                
            # Breaking gracefully
            if cv2.waitKey(5) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                quit()
