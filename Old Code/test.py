"""
File:   test.py
Author: Anthony Khoury and Keegan Bess
Date:   04/18/2022
Desc:    
"""

#%%
# Imports and Functions
from train import Train, GetPics, Image_to_EHD
import numpy as np
from numpy import load
from natsort import natsorted
import glob


print("\n------ Train ------")
# Input Vector
X = './train'

# Train on Input Data Set, X
knn, X_train, y_train = Train(folder=X)




#%%
print("\n------ Test ------")
def Test(knn, X_train, y_train, images):

     # Get List of Untrained Images and Truth Values
     # untrained_images, untrained_labels = GetPics(untrained_folder)
     untrained_images = images

     # Convert Images to EHD
     untrained_data = Image_to_EHD(untrained_images)

     # List -> Numpy Array
     untrained_data = np.array(untrained_data)
     # untrained_labels = np.array(untrained_labels)

     # Fit the Data
     fitted_knn = knn.fit(X_train, y_train)

     # Predict Value and Check if it's Correct
     for i in range(len(untrained_images)):
          # Predict Image Value
          predicted = fitted_knn.predict([untrained_data[i]])
          Y.append(predicted[0])

# Ouptut Vector
Y = []
# untrained_images = './Final_set'

untrained_images = []
path = r'C:\Users\antho\Documents\Academic\1. Spring 2022\3. Intro to Machine Learning\2. Project Assignments\Final Project\Final_Project_Beluga_Whales\Final_set\\'
for infile in natsorted(glob.glob(path + '*.jpg')):
     temp = str(infile)
     temp = './Final_set/' + str(temp.split('\\')[-1])
     untrained_images.append(temp)

Test(knn, X_train, y_train, untrained_images)

print("\tOutput Vector: ", Y)
np.save("Output.npy", Y)

#%%
data = load('Output.npy')
print(data)
print(len(data))







#%%
truth = [5,14,9,5,12,1,4,12,7,8,12,6,2,12,2,6,11,8,1,12,0,0,12,6,4,11,14,4,12,7,4,12,1,5,14,8,1,10,3,14,11,4,0,12,9,2,11,14,1,14,2,5,14,5,5,13]
correct = 0
for i in range(len(truth)):
     if truth[i] == Y[i]:
          correct += 1
     print('Actual: ' + str(truth[i]), ' | Predicted: ', str(Y[i]))
print(len(truth))
print("Percentage Correct: ", correct / len(truth))



# %%
