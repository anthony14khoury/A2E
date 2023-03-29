import threading
import cv2
import mediapipe as mp
from parameters import Params, extract_hand_keypoints
from keras.models import load_model
import numpy as np
import time


# Parameter Class
params = Params()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(model_complexity=0, max_num_hands=2)

# ML Model
try:
    model = load_model("./Models/128_full_tanh_model.h5")
except:
    model = load_model("./A2E/CameraTracking/Models/128_full_tanh_model.h5")
letters = params.LETTERS


# Define the function to process each image
def process_image(image):
     # with mp.solutions.hands.Hands() as hands:
     #      results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
     #      # Do something with the results, e.g. extract hand landmarks
     #      landmarks = results.multi_hand_landmarks
     
     # return landmarks

     image.flags.writeable = False
     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
     results = hands.process(image)

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



# Open the video capture device
cap = cv2.VideoCapture(0)

# Loop through the frames
FRAME_STORE = []
frames = []
with mp_hands.Hands(model_complexity=0, max_num_hands=2) as hands:
          
     while cap.isOpened():
          
          t1 = time.time()
          for frame_num in range(params.FRAME_COUNT):

               # Capture a frame
               success, image = cap.read()
               
               FRAME_STORE.append(image)
               
               time.sleep(0.075)
               
          print("Get 30 frames with delay: ", time.time() - t1)
               
          threads = []
          results = []
          for image in FRAME_STORE:
               t = threading.Thread(target=lambda r, i: r.append(process_image(i)), args=(results, image))
               threads.append(t)
               t.start()
          for t in threads:
               t.join()
          
          total_results = []
          for landmarks in results:
               total_results.append(landmarks)
          total_results = np.array(total_results)          
          
          prediction = model.predict(np.expand_dims(total_results, axis=0))
          char_index = np.argmax(prediction)
          confidence = round(prediction[0,char_index]*100, 1)
          predicted_char = letters[char_index]
          print(predicted_char, confidence)
          
          FRAME_STORE = []
          frames = []
          
          print("Total time taken: {}".format(time.time() - t1))