import numpy as np

class Params():
     def __init__(self):
          self.FRAME_STORE = []
          self.SEQUENCE_STORE = []
          self.EMPTY_HAND = [0] * 72
          self.FRAME_COUNT = 30
          self.SEQUENCE_COUNT = 20
          self.COLLECTION_FOLDER = 'DataCollection'
