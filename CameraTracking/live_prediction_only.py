from parameters import Params, extract_hand_keypoints
from keras.models import load_model
import mediapipe as mp
import numpy as np
import time
import cv2
import socket
# from word_detection import add_spaces

# Parameter Class
params = Params()

# ML Model
try:
    model = load_model("./Models/96_full_tanh_model.h5")
except:
    model = load_model("./A2E/CameraTracking/Models/128_26_15_model_tanh.h5")
letters = params.LETTERS

# Mediapipe Modules
mp_hands = mp.solutions.hands


# Socket Settings
HOST = "10.136.49.55" # The server's hostname or IP address
PORT = 4000 # The port used by the server



# Use CV2 Functionality to create a Video Stream
cap = cv2.VideoCapture(2)
# cap = cv2.VideoCapture(0)
while not cap.isOpened():
    pass
print("Camera is connected")


with mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=2) as hands:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        # connect to socket
       # s.connect((HOST, PORT))
        # Stay open while the camera is activated
        while cap.isOpened():

            FRAME_STORE = []
            IMAGE_STORE = []
            hands_count = 0
            t = time.time()
            for frame_num in range(params.FRAME_COUNT):
                success, image = cap.read()
                IMAGE_STORE.append(image)
                time.sleep(.05)
            print("recording took: ", time.time()-t)
            # Loop through all of the frames
            t0 = time.time()
            for frame_num in range(params.FRAME_COUNT):
                print(frame_num)
                t1 = time.time()
                # Capture a frame
                image = IMAGE_STORE[frame_num]
                #print("image read time:", time.time()-t1)
                # Error Checking
                # if not success:
                #     print("Ignoring Empty Camera Frame")
                #     continue

                # Make detections
                image.flags.writeable = False
                t2 = time.time()
                results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
               # print("hands.process took: ", time.time()-t2)

                if results.multi_handedness != None:
                    hands_count += len(results.multi_handedness)

                # Draw the Detections on the hand (comment out for PI predictions)
                #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # Don't need this if you are drawing landmarks
                #image = draw_styled_landmarks(image, results)
                t3 = time.time()
                FRAME_STORE.append(extract_hand_keypoints(results))
               # print("extract keypoints took:", time.time()-t3)
                # Display Image
                #cv2.imshow('WindowOutput', image)

                # Breaking gracefully
                #if cv2.waitKey(5) & 0xFF == ord('q'):
                    #cap.release()
                    #cv2.destroyAllWindows()
                   # quit()

            if hands_count > 0:
                prediction = model.predict(np.expand_dims(FRAME_STORE, axis=0))
                char_index = np.argmax(prediction)
                confidence = round(prediction[0,char_index]*100, 1)
                predicted_char = letters[char_index]
               # s.send(predicted_char)
                print(predicted_char, confidence)
            
            else:
                print("Nothing")
            print("total prediction time:", time.time()-t0)
            print("sleeping")
            time.sleep(2)
            print("go")
