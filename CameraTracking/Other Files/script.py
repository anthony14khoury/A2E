import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

### Keypoints using MP Holistic
def mediapipe_detection(image, model):
     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Color conversion from BGR -> RGB
     image.flags.writeable = False                  # Image is no longer writable
     results = model.process(image)                 # Make Prediction
     image.flags.writeable = True                   # Image is now writable
     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # Color conversion from RGB -> BGR
     return image, results

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections


def draw_styled_landmarks(image, results):
     # Draw left hand connections
     mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                              mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                              mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                              ) 
     # Draw right hand connections  
     mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                              mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                              mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                              ) 
     
     
### Extract Keypoint Values
# Stores all the keypoint values in an array
# Error checking if the hand isn't there, it'll insert a zero array
def extract_keypoints(results):
     lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
     rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
     return np.concatenate([lh, rh])


### Setup Folders for Collection

# Path for exported data, numpy arrays
DATA_PATH = os.path.join('New_Data')

# Actions that we try to detect
actions = np.array(['a', 'b', 'c'])

# Thirty videos worth of data
no_sequences = 30

# Videos are going to be 30 frames in length
sequence_length = 20

     
for action in actions: 
    for sequence in range(no_sequences):
        try: 
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass
       
       
### Collect Keypoint Values for Training & Testing     
cap = cv2.VideoCapture(1)

# Set mediapipe model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
     
     # Loop through actions
     for action in actions:
          
          # Loop through sequences (videos)
          for sequence in range(no_sequences):
               
               # Loop through video length (sequence length)
               for frame_num in range(sequence_length):

                    # Read Feed
                    ret, frame = cap.read()
                    
                    # Made detections
                    image, results = mediapipe_detection(frame, holistic)
                     
                    # Draw landmarks
                    draw_styled_landmarks(frame, results)
                    
                    # NEW Apply wait logic
                    if frame_num == 0: 
                         cv2.putText(image, 'STARTING COLLECTION', (120,200), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                         cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                         # Show to screen
                         cv2.imshow('OpenCV Feed', image)
                         cv2.waitKey(1000)
                    else: 
                         cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                         # Show to screen
                         cv2.imshow('OpenCV Feed', image)
                    
                    # NEW Export keypoints
                    keypoints = extract_keypoints(results)
                    npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                    np.save(npy_path, keypoints)
                
                    # Show to screen
                    cv2.imshow('OpenCV Feed', frame)
                    
                    # Breaking gracefully
                    if cv2.waitKey(5) & 0xFF == ord('q'):
                         break

cap.release()
cv2.destroyAllWindows()


### Preprocess Data and Create Lables and Features

from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
label_map = {label:num for num, label in enumerate(actions)}
sequences, labels = [], []
# Loop through actions

for action in actions:
     # Go through each number number in actions
     for sequence in np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int):
         
          window = []
          # Grab all 30 frames and append them to window
          for frame_num in range(sequence_length):
               res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
               window.append(res)
          
          sequences.append(window)
          labels.append(label_map[action])
          

X = np.array(sequences)
y = to_categorical(labels).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)


### Build and Train LSTM Neural Network

from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard
from tensorflow import keras

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

model = Sequential()

model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(20,126)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.fit(X_train, y_train, epochs=2000, callbacks=[tb_callback])