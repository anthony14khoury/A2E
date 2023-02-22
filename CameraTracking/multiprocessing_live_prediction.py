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

# Use CV2 Functionality to create a Video stream and add some values
cap = cv2.VideoCapture(0, cv2.CAP_ANY)

num_of_reader_procs = 4

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
letters = np.array(['a', 'b', 'c', 'e', 'f', 'j', 'nothing'])
model = load_model('abcefjnothing2.h5')

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


def process_frames(qq, FRAME_STORE, DRAWING_STORE):
    while True:
        frame = qq.get()
        # Produces the hand framework overlay ontop of the hand, you can choose the colour here too)
        if frame is None:
            continue

        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # In case the system sees multiple hands this if statment deals with that and produces another hand overlay
        keypoints = extract_keypoints(results)
        FRAME_STORE.put(keypoints)
        # draw landmarks
        if results.multi_hand_landmarks != None:
            for handLandmarks in results.multi_hand_landmarks:
                drawingModule.draw_landmarks(frame, handLandmarks, handsModule.HAND_CONNECTIONS)
        DRAWING_STORE.put(frame)


def predictor(qq):
    while True:
        for frame_num in range(params.FRAME_COUNT):
            ret, frame = cap.read()
            frame = cv2.resize(frame, (640, 480))
            qq.put(frame)
        time.sleep(2)


def output(FRAME_STORE, DRAWING_STORE):
    temp = []
    while True:
        currFrame = FRAME_STORE.get()
        currDrawing = DRAWING_STORE.get()
        if currFrame is None:
            continue
        temp.append(currFrame)

        if currDrawing is not None:
            cv2.imshow("Frame", currDrawing)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                quit()

        if len(temp) == params.FRAME_COUNT:
            prediction = model.predict(np.expand_dims(temp, axis=0))
            char_index = np.argmax(prediction)
            confidence = round(prediction[0, char_index] * 100, 1)
            predicted_char = letters[char_index]
            print(predicted_char, confidence)
            temp = []


def start_reader_procs(qq, FRAME_STORE,DRAWING_STORE, num_of_reader_procs):
    """Start the reader processes and return all in a list to the caller"""
    all_reader_procs = list()
    for ii in range(0, num_of_reader_procs):
        reader_p = multiprocessing.Process(target=process_frames, args=(qq, FRAME_STORE,DRAWING_STORE))
        reader_p.daemon = True
        all_reader_procs.append(reader_p)

    return all_reader_procs


if __name__ == '__main__':
    qq = multiprocessing.Queue() # cv2 frames
    FRAME_STORE = multiprocessing.Queue() # keypoints
    DRAWING_STORE = multiprocessing.Queue() # final image

    # handle generating keypoints and final image
    all_reader_procs = start_reader_procs(qq, FRAME_STORE,DRAWING_STORE, num_of_reader_procs)

    # gets raw frame from cv2
    producer_process = multiprocessing.Process(target=predictor, args=((qq),))

    # gets keypoints and final image and makes prediction and outputs image
    output_process = multiprocessing.Process(target=output, args=(FRAME_STORE, DRAWING_STORE))

    for idx, a_reader_proc in enumerate(all_reader_procs):
        a_reader_proc.start()
    producer_process.start()
    output_process.start()

    producer_process.join()
    output_process.join()
    for idx, a_reader_proc in enumerate(all_reader_procs):
        a_reader_proc.join()
