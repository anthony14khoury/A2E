from parameters import Params, extract_hand_keypoints
from keras.models import load_model
import mediapipe as mp
import numpy as np
import time
import os
import cv2
import socket
# from word_detection import add_spaces

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Parameter Class
params = Params()

# ML Model
try:
    model = load_model("./Models/96_tanh_model.h5")
except:
    model = load_model("./A2E/CameraTracking/Models/96_tanh_model.h5")
letters = params.LETTERS

# Mediapipe Modules
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Constants for Prediction Script
getReady = "Prepare to Sign!"
go = "Go!"
font = cv2.FONT_HERSHEY_SIMPLEX
color = (255,255,255)
fontScale = 1
thickness = 2



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
cap = cv2.VideoCapture(0, cv2.CAP_ANY)
# cap = cv2.VideoCapture(0)
while not cap.isOpened():
    pass
print("Camera is connected")


cv2.namedWindow("WindowOutput")
with mp_hands.Hands(model_complexity=0, min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=2) as hands:
    # Stay open while the camera is activated
    while cap.isOpened():

        FRAME_STORE = []
        t1 = time.time()
        print("Go")
        # Loop through all of the frames
        for frame_num in range(params.FRAME_COUNT):

            # Capture a frame
            success, image = cap.read()

            # Error Checking
            if not success:
                print("Ignoring Empty Camera Frame")
                continue

            # Make detections
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            # Draw the Detections on the hand (comment out for PI predictions)
            # image.flags.writeable = True
            # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # Don't need this if you are drawing landmarks
            image = draw_styled_landmarks(image, results)
            FRAME_STORE.append(extract_hand_keypoints(results))

            # Display Image
            cv2.imshow('WindowOutput', image)

            # Breaking gracefully
            if cv2.waitKey(5) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                quit()
                
            time.sleep(0.04)
        
        print("Time taken: ", time.time() - t1)

        prediction = model.predict(np.expand_dims(FRAME_STORE, axis=0))
        char_index = np.argmax(prediction)
        confidence = round(prediction[0,char_index]*100, 1)
        predicted_char = letters[char_index]
        print(predicted_char, confidence)
        
        print("Wait 2 second")
        time.sleep(2.0)


        # """ Continuous Camera Share """
        # timeout = time.time() + 1
        # while time.time() < timeout:

        #     # Read Feed
        #     success, image2 = cap.read()

        #     image.flags.writeable = False
        #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        #     # image.flags.writeable = True
        #     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # Don't need this if you are drawing landmarks

        #     cv2.imshow('WindowOutput', image2)