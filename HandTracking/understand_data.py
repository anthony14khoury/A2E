import numpy as np

# data_frame = np.load("./DataCollection/nothing/nothing0.npy")
data_frame = np.load("./DataCollection/l/l0.npy")
for i in range(len(data_frame)):
     print(data_frame[i][170])
# print("Number of Hands: {}".format(data_frame[0][0]))
# print("Number of Fingers: {}".format(data_frame[0][1]))
# print("Left Hand Data: ")

# for i in range(2, 5):
#      print("\t Hand Palm Normal ({}): {}".format(i-2, data_frame[0][i]))

# for i in range(5, 8):
#      print("\t Hand Direction ({}): {}".format(i-5, data_frame[0][i]))
     
# print("Hand Direction Pitch: ", data_frame[0][8])
     
