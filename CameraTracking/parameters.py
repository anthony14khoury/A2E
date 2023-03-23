import numpy as np
import cv2

class Params():
     def __init__(self):
          self.FRAME_STORE = []
          self.SEQUENCE_STORE = []
          self.FRAME_COUNT = 30
          self.SEQUENCE_COUNT = 25
          self.LETTERS = np.array(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'nothing', 'again', 'can', 'drink', 'family', 'hello', 'how are you', 'me', 'my', 'name is', 'nice to meet you', 'no', 'please', 'sorry', 'thank you', 'yes'])


def mediapipe_detection(image, model):
     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Color conversion from BGR -> RGB
     image.flags.writeable = False                  # Image is no longer writable
     results = model.process(image)                 # Make Prediction
     image.flags.writeable = True                   # Image is now writable
     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # Color conversion from RGB -> BGR
     return image, results



def extract_hand_keypoints(results):
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



def extract_holistic_keypoints(results):
     
     if results.left_hand_landmarks: 
          left_hand = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten()
     else: 
          left_hand = np.zeros(21*3)
          
     if results.right_hand_landmarks: 
          rignt_hand = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten()
     else: 
          rignt_hand = np.zeros(21*3)
          
     return np.concatenate([left_hand, rignt_hand])
