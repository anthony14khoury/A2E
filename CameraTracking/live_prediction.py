from parameters import Params, extract_hand_keypoints
from keras.models import load_model
# import mediapipe as mp
from mediapipe import solutions
import numpy as np
import time
import cv2
import socket
# from word_detection import add_spaces

# Parameter Class
params = Params()

# ML Model
try:
    model = load_model("./Models/128_26_15_model_tanh.h5")
except:
    model = load_model("./A2E/CameraTracking/Models/128_26_15_model_tanh.h5")
letters = params.LETTERS

# Mediapipe Modules
mp_drawing = solutions.drawing_utils
mp_drawing_styles = solutions.drawing_styles
mp_hands = solutions.hands

# Constants for Prediction Script
getReady = "Prepare to Sign!"
go = "Go!"
font = cv2.FONT_HERSHEY_SIMPLEX
color = (255,255,255)
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
cap = cv2.VideoCapture(0, cv2.CAP_ANY)
# cap = cv2.VideoCapture(0)
while not cap.isOpened():
    pass
print("Camera is connected")


with mp_hands.Hands(model_complexity=0, min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=2) as hands:
    # with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        # connect to socket
        # s.connect((HOST, PORT))
        # Stay open while the camera is activated
    while cap.isOpened():
        t1 = time.time()
        FRAME_STORE = []
        # hands_count = 0

        # Loop through all of the frames
        for frame_num in range(params.FRAME_COUNT):

            # Capture a frame
            success, image = cap.read()
            
            FRAME_STORE.append(image)

            # Error Checking
            # if not success:
            #     print("Ignoring Empty Camera Frame")
            #     continue

            # Make detections
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            # if results.multi_handedness != None:
            #     hands_count += len(results.multi_handedness)

            # Draw the Detections on the hand (comment out for PI predictions)
            # image.flags.writeable = True
            # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # Don't need this if you are drawing landmarks
            # image = draw_styled_landmarks(image, results)
            FRAME_STORE.append(extract_hand_keypoints(results))

            # Display Image
            # cv2.imshow('WindowOutput', image)

            # Breaking gracefully
            # if cv2.waitKey(5) & 0xFF == ord('q'):
            #     cap.release()
            #     cv2.destroyAllWindows()
            #     quit()

        # if hands_count > 0:
        prediction = model.predict(np.expand_dims(FRAME_STORE, axis=0))
        char_index = np.argmax(prediction)
        confidence = round(prediction[0,char_index]*100, 1)
        predicted_char = letters[char_index]
        # s.send(predicted_char)
        print(predicted_char, confidence)
        print("Total time taken: {}".format(time.time() - t1))

            # else:
            #     print("Nothing")


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

                # Show to screen
                # image = cv2.putText(image, getReady, (int(len(image[0])/2)-200, int(len(image)/2)), font, fontScale, color, thickness, cv2.LINE_AA)
                # cv2.imshow('OpenCV Feed', image)



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