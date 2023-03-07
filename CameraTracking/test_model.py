from parameters import Params
from sklearn.metrics import confusion_matrix
from keras.models import load_model
import mediapipe as mp
import numpy as np
import cv2
import os


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
     

collection_folder = 'HandsCollection'
skips = 10
skip_cout = 0
captured_count = 0

debug = True
if debug:
     collection_folder = "./A2E/CameraTracking/" + collection_folder

params = Params()
letters = params.LETTERS
# test_letters = ['b']
test_letters = letters
labels = []

letter_count = 0
bool = True
for subdir, dirs, files in os.walk(collection_folder):
     if bool: letter_count = len(dirs)
     bool = False
     
     if subdir[-1] in test_letters:
          for file in files:
               labels.append(file[0])


          
          
sequences = []
for letter in test_letters:
     try:
          dir_length = len(os.listdir(os.path.join(collection_folder, letter)))
          
          for i in range(0, dir_length):
               sequences.append(np.load(os.path.join(collection_folder, letter, letter + str(i) + ".npy")))
               # print(os.path.join(collection_folder, letter, letter + str(i) + ".npy"))
          
     except:
          # print("No validation test for letter: {}".format(letter))
          dummy = 0



model = load_model("./A2E/CameraTracking/Models/128_model_tanh_6.h5")
     
predictions = []
confidences = []
for (label, sequence) in zip(labels, sequences):
     prediction = model.predict(np.expand_dims(sequence, axis=0), verbose=0)
     char_index = np.argmax(prediction)
     confidence = round(prediction[0,char_index]*100, 1)
     predicted_char = letters[char_index]
     
     print("Actual: {} | Predicted: {} | Confidence: {}".format(label, predicted_char, confidence))
     
     predictions.append(predicted_char)
     confidences.append(confidence)

accuracy = sum(1 for x,y in zip(labels, predictions) if x == y) / len(labels)
print("Total Accuracy: {}".format(accuracy))

cm = confusion_matrix(labels, predictions)

# Accuracy per Label
print()
acc_per_label = {}
for i in range(letter_count): # change it to letter_count if you're testing all the letters
     label_total = sum(cm[i, :])
     label_correct = cm[i, i]
     print("Letter: {} | Accuracy: {}".format(test_letters[i], label_correct / label_total))
          