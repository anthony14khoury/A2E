# Import the necessary Packages for this software to run
import mediapipe
import cv2
import numpy as np
from keras.models import load_model
from parameters import Params
import time as time

# Use MediaPipe to draw the hand framework over the top of hands it identifies in Real-Time
drawingModule = mediapipe.solutions.drawing_utils
handsModule = mediapipe.solutions.hands

# Use CV2 Functionality to create a Video stream and add some values
cap = cv2.VideoCapture(0, cv2.CAP_ANY)

# Wait until camera is connected
while not cap.isOpened():
    pass
print("Camera is connected")

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
letters = np.array(['a', 'b', 'c', 'e', 'f', 'j', 'nothing'])
model = load_model('abcefjnothing2.h5')
params = Params()

def extract_keypoints(results):
    lh = np.zeros(21*3)
    rh = np.zeros(21*3)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get hand index to check label (left or right)
            handIndex = results.multi_hand_landmarks.index(hand_landmarks)
            handLabel = results.multi_handedness[handIndex].classification[0].label
            if(handLabel == "Right"):
                lh = np.array([[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks.landmark]).flatten()
            else:
                rh = np.array([[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks.landmark]).flatten()
    return np.concatenate([lh, rh])


with handsModule.Hands(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7,
                       max_num_hands=2) as hands:
    # Create an infinite loop which will produce the live feed to our desktop and that will search for hands
    while True:
        FRAME_STORE = []
        for frame_num in range(params.FRAME_COUNT):
            ret, frame = cap.read()
            start = time.time()
            # Unedit the below line if your live feed is produced upsidedown
            # flipped = cv2.flip(frame, flipCode = -1)
            frame = cv2.resize(frame, (640, 480))
            # Determines the frame size, 640 x 480 offers a nice balance between speed and accurate identification

            # Produces the hand framework overlay ontop of the hand, you can choose the colour here too)
            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # In case the system sees multiple hands this if statment deals with that and produces another hand overlay
            keypoints = extract_keypoints(results)
            FRAME_STORE.append(keypoints)

            # draw landmarks
            #if results.multi_hand_landmarks != None:
             #   for handLandmarks in results.multi_hand_landmarks:
              #      drawingModule.draw_landmarks(frame, handLandmarks, handsModule.HAND_CONNECTIONS)
            print(time.time()-start)
            # Below shows the current frame to the desktop
            #cv2.imshow("Frame", frame)
            #key = cv2.waitKey(1) & 0xFF
            # Below states that if the |q| is press on the keyboard it will stop the system

        prediction = model.predict(np.expand_dims(FRAME_STORE, axis=0))
        char_index = np.argmax(prediction)
        confidence = round(prediction[0,char_index]*100, 1)
        predicted_char = letters[char_index]
        print(predicted_char, confidence)
        time.sleep(2)
