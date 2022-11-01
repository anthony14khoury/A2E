# Final_Project_Beluga_Whales
By: Anthony Khoury and Keegan Bess

## How to run the code:
---
- In the root folder of this repo, add a folder with images to predict
- On line 48 of `test.py`, there is a variable called "untrained_images". Switch value of the variable to the name of the folder added
- Run the following command in the terminal or click "Run Below" using Jupyter Notebook at the top of the `test.py`
     - `python test.py`

## About the Project
---
There are 2 folders with images:
- `train` contains all the images to train the algorithm
- `untrained` contains all the images that will be predicted

There are 3 Python Scripts:
- `EHD.py` receives an image and returns a feature vector as an Edge Histogram Descriptor
- `train.py` receives a 'train' folder with a list of images and performs Adaptive Gaussian Thresholding, EHD, and KNN with cross-validation for all the images in the untrained folder.
- `test.py` receives and untrained folder with images to predict and returns a vector "Y" with predicted values
