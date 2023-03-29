from parameters import Params
from keras.models import load_model
import mediapipe as mp
from mediapipe import solutions
import asyncio
import numpy as np
import time
import cv2

# Parameter Class
params = Params()
letters = params.LETTERS

# ML Model
try:
    model = load_model("./Models/128_26_15_model_tanh.h5")
except:
    model = load_model("./A2E/CameraTracking/Models/128_26_15_model_tanh.h5")

# Mediapipe Modules
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(model_complexity=0, max_num_hands=2)


async def extract_hand_keypoints(result):
       
    lh = np.zeros(21*3)
    rh = np.zeros(21*3)
    
    result.flags.writeable = False
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    result = hands.process(result)
        
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Get hand index to check label (left or right)
            handIndex = result.multi_hand_landmarks.index(hand_landmarks)
            handLabel = result.multi_handedness[handIndex].classification[0].label
            if(handLabel == "Right"):
                lh = np.array([[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks.landmark]).flatten()
            else:
                rh = np.array([[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks.landmark]).flatten()
            
    return np.concatenate([lh, rh])

# Use CV2 Functionality to create a Video Stream
cap = cv2.VideoCapture(2, cv2.CAP_ANY)
while not cap.isOpened():
    pass
print("Camera is connected")

async def main():
    
    FRAME_STORE = []
    frames = []
    with mp_hands.Hands(model_complexity=0, max_num_hands=2) as hands:
            
        while cap.isOpened():
            
            t1 = time.time()
            for frame_num in range(params.FRAME_COUNT):

                # Capture a frame
                success, image = cap.read()
                
                FRAME_STORE.append(image)
                
                if len(FRAME_STORE) == 30:
                    
                    # Run the detect_hands coroutine asynchronously
                    for frame in FRAME_STORE:
                        frames.append(await extract_hand_keypoints(np.array(frame)))
                    
                    frames = np.array(frames)
                    prediction = model.predict(np.expand_dims(frames, axis=0))
                    char_index = np.argmax(prediction)
                    confidence = round(prediction[0,char_index]*100, 1)
                    predicted_char = letters[char_index]
                    print(predicted_char, confidence)
                    
                    FRAME_STORE = []
                    frames = []
                    print("Total time taken: {}".format(time.time() - t1))
            
asyncio.run(main())
