# ASL to English (A2E)

## Purpose

- This application aims to aid individuals with hearing impairments to communicate with non-asl speakers with ease. Due to the lack of translation resources, hearing-impaired individuals are restricted in what they can do and who they can talk to. A2E removes this barrier.

## Sections
- Machine Learning Model
    - The current model is trained with the following letters (26) and words (16):

    |         |                  |       |        |
    | :---:   | :---:            | :---: | :---:  |
    |  a      | b                | c     | d      |
    |  e      | f                | g     | h      |
    |  i      | j                | k     | l      |
    |  m      | n                | o     | p      |
    |  q      | r                | s     | t      |
    |  u      | v                | w     | x      |
    |  y      | z                |       |        |
    | again   | can              | drink | family |
    | hello   | how are you      | me    | my     |
    | name is | nice to meet you | no    | please |
    | sorry    | thank you       | yes   | nothing      |
    
    - The model's average prediction accuracy is 96.3%

- Android Application
    - The user will sign through to the external camera and the application will output the camera stream and predictions in real-time.

- Raspberry Pi
    - All of the code and the machine learning model is loaded on to the PI to communicate directly to the application using TCP for the predictions and HTTP for the video stream

## Folders
- CameraTracking
    - Contains all updated code, data, and models

- Socket
    - Code to connect Raspberry Pi and Application

- Old Folders
    - Leap Image Model & Leap Tracking Models contain code, data, and models for using the LEAP controller

## Dependencies to Install
- Python 3.7
    - cv2
    - numpy
    - matplotlib
    - mediapipe == 0.8.9.1
    - keras
    - sklearn
    - PyEnchant
    - wordfreq
    - pandas
- Once all dependencies are installed, run the ```python_check.py``` script to verify everything is properly installed.


## Important Files in Camera Tracking
- data_collection.py
    - This script collects hand-tracked data through mediapipe for a specified letter or word.
    - Each run will collect 20 sequences and each sequence contains 30 frames which allows motion to be incorporated into our model.
    - Sequences are saved as npy files in the 'DataCollection' folder

- live_prediction.py
    - This script opens a camera to sign to and based on the motion of your hand, it will output a corresponding letter or word with the predicted accuracy.

- model.py
    - Gathers all of the data in a structured manner to input into a custom neural network
    - Outputs an 'h5' file which has the fully created model

- parameters.py
    - A class with consistent values to be used throughout the code

## How to run the code
1. Navigate to the "CameraTracking" folder
2. Run the following command
    - `python live_prediction.py`
3. That's it, you should see a predicted letter or word and the model's confidence score for it.
