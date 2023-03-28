import threading
import cv2
import mediapipe as mp
from parameters import Params, extract_hand_keypoints
from keras.models import load_model
import numpy as np


# Parameter Class
params = Params()
FRAME_STORE = []

# ML Model
try:
    model = load_model("./Models/128_26_15_model_tanh.h5")
except:
    model = load_model("./A2E/CameraTracking/Models/128_26_15_model_tanh.h5")
letters = params.LETTERS



# Define a function to process each frame
def process_frame(frame):
     # Initialize mediapipe
     mp_hands = mp.solutions.hands.Hands()

     # Convert the frame to RGB
     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

     # Process the frame
     results = mp_hands.process(frame)
     FRAME_STORE.append(extract_hand_keypoints(results))

     # Print the results
     print(results)

     # Release the resources
     mp_hands.close()





# Open the video capture device
cap = cv2.VideoCapture(0)

# Loop through the frames
count = 0
while True:
     # Read a frame from the video capture device
     ret, frame = cap.read()
     if not ret:
          break
     count += 1
     
     if count >= 30:
          prediction = model.predict(np.expand_dims(FRAME_STORE, axis=0))
          char_index = np.argmax(prediction)
          confidence = round(prediction[0,char_index]*100, 1)
          predicted_char = letters[char_index]
          # s.send(predicted_char)
          print(predicted_char, confidence)
          FRAME_STORE = []

     # Create a thread to process the frame
     thread = threading.Thread(target=process_frame, args=(frame,))
     thread.start()

     # Wait for the thread to finish
     thread.join()

     # Show the frame
     # cv2.imshow('frame', frame)

     # Check for key press
     # if cv2.waitKey(1) == ord('q'):
     #      break

# Release the video capture device and destroy the windows
cap.release()
cv2.destroyAllWindows()
