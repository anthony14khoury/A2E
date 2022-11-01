"""
File:   train.py
Author: Anthony Khoury and Keegan Bess
Date:   04/18/2022
Desc:   
"""

#%% 
# Imports & Functions
from EHD import EHD_Parameters, Edge_Histogram_Descriptor
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import os
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import rand_score
from sklearn import neighbors



def KNN(n, X_train, y_train, X_val, y_val, distance_metric):
     # Purpose: Perform KNN
    
     #Run KNN
     neighbor = KNeighborsClassifier(n_neighbors=n, metric=distance_metric)
     
     #Fit the KNN with the training data
     neighbor.fit(X_train,y_train)
     
     #Obtain prediction based off trained KNN
     pred = neighbor.predict(X_val)
     
     #compute rand index between two clusterings, true vs prediction
     return rand_score(y_val, pred)

def Image_to_EHD(image_list):
     # Purpose: Convert Images to EHD's and train them

     print("\tNumber of Images being trained: ", len(image_list))

     # Looping through all of the images
     data = np.empty((0, 9))

     for i in range(len(image_list)):
     # for i in range(5):

          if i % 100 == 0: print("\tImages trained: ", i)

          # Images
          image = cv2.imread(image_list[i], 0)
          altered_image = cv2.medianBlur(image, 3)
          altered_image = cv2.adaptiveThreshold(altered_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 101, 14)
          altered_image = Image.fromarray(altered_image)


          # Image -> Tensor
          transform = transforms.Compose([transforms.PILToTensor()])
          image_tensor = transform(altered_image)
          image_tensor1 = image_tensor.unsqueeze(0).float()

          # Calculating the Edge Histogram Descriptors
          feature_vector = Edge_Histogram_Descriptor(image_tensor1, EHD_Parameters, None)

          # Modifying the Dimensions from 4D -> 3D
          cube = np.array(feature_vector[0])
          angles = ['0', '45', '90', '135', '180', '225', '270', '315', 'None']
          angle_values = []
          for i in range(cube.shape[0]):
               sum = 0
               for j in range(cube[i].shape[0]):
                    sum += int(np.array(cube[i][j]).sum()) # Adding up the percentages of each pixel
               angle_values.append(sum)   

          # Store output data
          data = np.append(data, np.array([angle_values]), axis=0)
     
     return data

def GetPics(folder):
     # Purpose: Loop through folder to return an array of all the images

    image_list = []
    labels_list = []
    for subdir, dirs, files in os.walk(folder):
        for file in files:
            
            label = file[0]
            if   (label == 'a' or label == 'A'): labels_list.append(10)
            elif (label == 's' or label == 'S'): labels_list.append(11)
            elif (label == 'm' or label == 'M'): labels_list.append(12)
            elif (label == 'd' or label == 'D'): labels_list.append(13)
            elif (label == 'r' or label == 'R'): labels_list.append(14)
            else: labels_list.append(int(label))

            image_list.append(os.path.join(subdir, file))
     
    return image_list, labels_list

def Train(folder):

     # Parameters:
     cv = 65
     neighbor = 1

     # Get List of Images and Truth Values
     images, labels = GetPics(folder)

     # Convert Images to EHD
     data = np.array(Image_to_EHD(images))
     labels = np.array(labels)
     labels = labels.reshape(labels.shape[0])

     #%% Split Data into Train and Test & Run KNN
     X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.20)

     # Calculate Accuracy Score of Trained Data
     score = []
     kfold = KFold(n_splits=cv, shuffle=True)
     for train_index, val_index in kfold.split(X_train):
          X1_train, X1_val = X_train[train_index], X_train[val_index]
          y1_train, y1_val = y_train[train_index], y_train[val_index]
          score += [KNN(neighbor, X1_train, y1_train, X1_val, y1_val, 'euclidean')]
     
     # Perform KNN
     knn = neighbors.KNeighborsClassifier(n_neighbors=neighbor)
     knn_scores = cross_val_score(knn, X_train, y_train, cv=cv, scoring='accuracy')
     max = np.max(knn_scores)
     print("\n\tKNN Accuracy Score: ", round(max, 4), "%")

     return  knn, X_train, y_train
# %%
