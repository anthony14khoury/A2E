# Import the necessary Packages for this software to run
import mediapipe
import cv2
import numpy as np
from keras.models import load_model
from parameters import Params
import time as time
import multiprocessing as multiprocessing

params = Params()
# Use MediaPipe to draw the hand framework over the top of hands it identifies in Real-Time
drawingModule = mediapipe.solutions.drawing_utils
handsModule = mediapipe.solutions.hands

num_of_reader_procs = 4

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
letters = np.array(['a', 'b', 'c', 'e', 'f', 'j', 'nothing'])
model = load_model('Models/Old/abcefjnothing2.h5')

def initialize():
    global hands
    hands = handsModule.Hands(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7,
                             max_num_hands=2)
def extract_keypoints(results):
    lh = np.zeros(21 * 3)
    rh = np.zeros(21 * 3)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get hand index to check label (left or right)
            handIndex = results.multi_hand_landmarks.index(hand_landmarks)
            handLabel = results.multi_handedness[handIndex].classification[0].label
            if (handLabel == "Right"):
                lh = np.array([[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks.landmark]).flatten()
            else:
                rh = np.array([[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks.landmark]).flatten()
    return np.concatenate([lh, rh])


def process_data(data):
    global hands
    results = hands.process(cv2.cvtColor(data, cv2.COLOR_BGR2RGB))
    # In case the system sees multiple hands this if statment deals with that and produces another hand overlay
    keypoints = extract_keypoints(results)
    return keypoints

def worker(data):
    return data

def my_generator():
    # Use CV2 Functionality to create a Video stream and add some values
    cap = cv2.VideoCapture(0, cv2.CAP_ANY)
    while True:
        num = 0
        while(num < 30):
            ret, frame = cap.read()
            frame = cv2.resize(frame, (640, 480))
            num = num + 1
            yield frame
        time.sleep(2)

if __name__ == '__main__':

    # create pools
    input_pool = multiprocessing.Pool(processes=4, initializer=initialize)
    output_pool = multiprocessing.Pool(processes=1)

    # continuously print resultsa
    num = 0
    temp = []
    for result in output_pool.imap(func=worker, iterable=input_pool.imap(func=process_data, iterable=my_generator())):
        num += 1
        temp.append(result)
        if(num == 30):
            prediction = model.predict(np.expand_dims(temp, axis=0))
            char_index = np.argmax(prediction)
            confidence = round(prediction[0, char_index] * 100, 1)
            predicted_char = letters[char_index]
            print(predicted_char, confidence)
            temp = []
            num = 0


