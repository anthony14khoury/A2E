import numpy as np
import cv2

class Params():
     def __init__(self):
          self.FRAME_STORE = []
          self.SEQUENCE_STORE = []
          # self.EMPTY_HAND = [0] * 72
          self.EMPTY_HAND = [0] * 120
          self.FRAME_COUNT = 30
          self.SEQUENCE_COUNT = 20
          self.COLLECTION_FOLDER = 'DataCollection'
          # self.LETTERS = np.array(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 'my', 'name', 'nothing'])
          # self.LETTERS = np.array(['a', 'e', 's', 'nothing'])
          self.LETTERS = np.array(['hello', 'my', 'name is', 'j', 'o', 'e', 'nothing'])


def mediapipe_detection(image, model):
     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Color conversion from BGR -> RGB
     image.flags.writeable = False                  # Image is no longer writable
     results = model.process(image)                 # Make Prediction
     image.flags.writeable = True                   # Image is now writable
     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # Color conversion from RGB -> BGR
     return image, results



def extract_keypoints(results):
     
     if results.left_hand_landmarks: 
          left_hand = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten()
     else: 
          left_hand = np.zeros(21*3)
          
     if results.right_hand_landmarks: 
          rignt_hand = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten()
     else: 
          rignt_hand = np.zeros(21*3)
          
     return np.concatenate([left_hand, rignt_hand])



""" 
Data Collection Informationq

Relevant Pose Landmarks (begins at 0)
11. Left Shoulder
12. Right Shoulder
13. Left Elbow
14. Right Elbow
15. Left Wrist
16. Right Wrist
17. Left Pinky
18. Right Pinky
19. Right Index
20. Left Thumb
21. Left Thumb
22. Right Thumb

Each landmark consists of the following
- x and y coordinates, normalized to [0.0, 1.0] by the image width and height respectively
- z = represents the landmark depth at the midpoint of hips being the origin and the smaller 
- visibility, a value in [0.0, 1.0] indicating the liklihood of the landmark being visible (present and not occluded) in the image.

0-4 4-9 9-13 13-17 17-21 21-25 25-29 29-33 33-37 37-41 41-45 45-49 49-53 53-57 57-61 61-65 65-69 69-73 73-77 77-81 81-85 85-89 89-93
 0    1  2     3     4     5     6     7     8     9    10    11    12    13   14     15    16     17   18   19     20   21     22

"""