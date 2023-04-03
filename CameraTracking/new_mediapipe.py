import mediapipe as mp
import cv2
import time
import numpy as np
from parameters import Params
from keras.models import load_model


BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Parameter Class
params = Params()

# ML Model
try:
    model = load_model("./Models/128_26_15_model_tanh.h5")
except:
    model = load_model("./A2E/CameraTracking/Models/128_26_15_model_tanh.h5")
letters = params.LETTERS

frames = []
wait = True

def extract_hand_keypoints(results):
    lh = np.zeros(21*3)
    rh = np.zeros(21*3)
    if results.handedness:
        count = 0
        for hand_landmarks in results.handedness:
            # Get hand index to check label (left or right)
            handIndex = count
            count += 1
            handLabel = hand_landmarks[0].category_name
            if(handLabel == "Right"):
                landmarks = results.hand_landmarks[handIndex]
                i = 0
                for set in landmarks:
                    lh[i] = set.x
                    lh[i+1] = set.y
                    lh[i+2] = set.z
                    i += 3
            else:
                landmarks = results.hand_landmarks[handIndex]
                i = 0
                for set in landmarks:
                    rh[i] = set.x
                    rh[i+1] = set.y
                    rh[i+2] = set.z
                    i += 3

    return np.concatenate([lh, rh])

# Create a hand landmarker instance with the live stream mode:
def print_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global frames
    global wait
    currFrame = extract_hand_keypoints(result)
    frames.append(currFrame)
    if len(frames) == 30:
        prediction = model.predict(np.expand_dims(frames, axis=0))
        char_index = np.argmax(prediction)
        confidence = round(prediction[0,char_index]*100, 1)
        predicted_char = letters[char_index]
        # s.send(predicted_char)
       # print(prediction)
        print(predicted_char, confidence)
        frames = []
        time.sleep(2)
        wait = False


if __name__ == '__main__':
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=print_result,
        num_hands=2)
    with HandLandmarker.create_from_options(options) as landmarker:
        cap = cv2.VideoCapture(0)
        while not cap.isOpened():
            pass
        t0 = int(time.time()*1000)
        while True:
            timer = time.time()
            print("go")
            for i in range(30):
                _, image = cap.read()
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
                currTimestamp = int(time.time()*1000) - t0
                landmarker.detect_async(mp_image, currTimestamp)
                time.sleep(0.05)
            print("stop")
            while wait:
                pass
            wait = True
            print("time taken:", time.time()-timer)

