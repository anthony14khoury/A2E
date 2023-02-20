import numpy as np

class Params():
     def __init__(self):
          self.FRAME_STORE = []
          self.SEQUENCE_STORE = []
          # self.EMPTY_HAND = [0] * 72
          self.EMPTY_HAND = [0] * 120
          self.FRAME_COUNT = 30
          self.SEQUENCE_COUNT = 20
          self.COLLECTION_FOLDER = 'Test'
          # self.LETTERS = np.array(['a', 'b', 'c', 'hello','j', 'n', 'name', 'nothing'])
          # self.LETTERS = np.array(['a', 'c', 'hello', 'name', 'nothing'])
          self.LETTERS = np.array(['down', 'nothing', 'side', 'up'])



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