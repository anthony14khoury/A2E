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
done = False
try:
    model = load_model("./Models/128_26_15_model_tanh.h5")
except:
    model = load_model("./A2E/CameraTracking/Models/128_26_15_model_tanh.h5")
letters = params.LETTERS

def initialize():
    global hands
    hands = handsModule.Hands(model_complexity=0,static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5,
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
    cap = cv2.VideoCapture(2)
    global done
    while True:
        num = 0
        t0 = time.time()
        while(num < 30):
            ret, frame = cap.read()
            frame = cv2.resize(frame, (640, 480))
            num = num + 1
            yield frame
        while(done == False):
            pass
        done = False
        print(time.time()-t0)

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
            time.sleep(2)
            done = True
            num = 0


