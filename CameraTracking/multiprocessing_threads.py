import threading
import cv2
import mediapipe as mp
from parameters import Params, extract_hand_keypoints
from keras.models import load_model
import numpy as np
import time
from queue import Queue


# Parameter Class
params = Params()
FRAME_STORE = []

# Define the number of threads
num_threads = 4

# Define the batch size
batch_size = 10

# Define a queue to hold the frames
frame_queue = Queue()

# ML Model
try:
    model = load_model("./Models/128_full_tanh_model.h5")
except:
    model = load_model("./A2E/CameraTracking/Models/128_full_tanh_model.h5")
letters = params.LETTERS



# Define a function to process each frame
def process_batch():
     print("Frame Count: ", len(FRAME_STORE))
     # Initialize mediapipe
     mp_hands = mp.solutions.hands.Hands(model_complexity=0, max_num_hands=2)
     
     while True:
          # Get the next batch of frames from the queue
          batch = []
          for i in range(batch_size):
               frame = frame_queue.get()
               if frame is None:
                    break
               batch.append(frame)

          # If the batch is empty, exit the loop
          if not batch:
               break

          # Convert the frames to RGB
          batch = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in batch]

          # Process the batch
          results = mp_hands.process(batch)
          
          for b in batch:
               FRAME_STORE.append(extract_hand_keypoints(b))

          # Print the results
          # print(results)

     # Release the resources
     mp_hands.close()


# Start the worker threads
threads = []
for i in range(num_threads):
    thread = threading.Thread(target=process_batch)
    thread.start()
    threads.append(thread)


# Open the video capture device
cap = cv2.VideoCapture(0)

# Loop through the frames
while True:
     t1 = time.time()
     # Read a frame from the video capture device
     ret, frame = cap.read()
     if not ret:
          break
     
     frame_queue.put(frame)
     # If the queue is full, wait for the worker threads to finish
     if frame_queue.qsize() == batch_size * num_threads:
          for i in range(num_threads):
               frame_queue.put(None)
          for thread in threads:
               thread.join()
     
     if len(FRAME_STORE) == 30:
          prediction = model.predict(np.expand_dims(FRAME_STORE, axis=0))
          char_index = np.argmax(prediction)
          confidence = round(prediction[0,char_index]*100, 1)
          predicted_char = letters[char_index]
          # s.send(predicted_char)
          print(predicted_char, confidence)
          print("time to predict: ", time.time() - t1)
          FRAME_STORE = []
          count = 0

     # Create a thread to process the frame
     # thread = threading.Thread(target=process_frame, args=(frame,))
     # thread.start()

     # Wait for the thread to finish
     # thread.join()

     # Show the frame
     # cv2.imshow('frame', frame)

     # Check for key press
     # if cv2.waitKey(1) == ord('q'):
     #      break

# Release the video capture device and destroy the windows
cap.release()
cv2.destroyAllWindows()
