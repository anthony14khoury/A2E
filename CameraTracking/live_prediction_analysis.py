from parameters import Params, extract_hand_keypoints
from keras.models import load_model
import numba as nb
import mediapipe as mp
import numpy as np
import time
import cv2
import os
from numba.typed import List
from numba import types
from mediapipe.python.solutions import hands
    


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# Parameter Class
params = Params()
# Mediapipe Modules
mp_hands = mp.solutions.hands


@nb.jit(target_backend='cuda')
def process_frames(IMAGE_STORE: types.List, hands: hands.Hands) -> types.List:
    FRAME_STORE = []
    for frame_num in range(30):
        # Capture a frame
        image = IMAGE_STORE[frame_num]
        image.flags.writeable = False
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        results1 = extract_hand_keypoints(results)
        FRAME_STORE.append(results1)
        
    return FRAME_STORE


# ML Model
try:
    model = load_model("./Models/96_tanh_model.h5")
except:
    model = load_model("./A2E/CameraTracking/Models/96_tanh_model.h5")
letters = params.LETTERS



# Use CV2 Functionality to create a Video Stream
cap = cv2.VideoCapture(0, cv2.CAP_ANY)
while not cap.isOpened():
    pass
print("Camera is connected")

with mp_hands.Hands(model_complexity=0, min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=2) as hands:
    
    # Stay open while the camera is activated
    while cap.isOpened():

        FRAME_STORE = []
        IMAGE_STORE = []

        t0 = time.time()
        
        print("Get Ready")
        time.sleep(1)
        print("Go")
        for frame_num in range(params.FRAME_COUNT):
            success, image = cap.read()
            IMAGE_STORE.append(image)
            cv2.imshow('OpenCV Feed', image)
            cv2.waitKey(1)
            time.sleep(.03)
        print("Recording took: ", time.time()-t0)
        
        # Loop through all of the frames
        t1 = time.time()
        FRAME_STORE = process_frames(IMAGE_STORE, hands)
        print("Process Frames: ", time.time()-t1)
        
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
