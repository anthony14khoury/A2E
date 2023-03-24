from parameters import Params
from sklearn.metrics import confusion_matrix, accuracy_score
from keras.models import load_model
import mediapipe as mp
import numpy as np
import cv2
import os
print("Import and Dependencies Loaded")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
     
skips = 10
skip_cout = 0
captured_count = 0

params = Params()
letters = params.LETTERS
test_letters = ['a', 'again', 'b', 'c', 'can', 'd', 'drink', 'e', 'f', 'family', 'g', 'h', 'hello', 'i', 'j', 'k', 'l', 'm', 'me', 'my', 'n', 'nothing', 'o', 'p', 'please', 'q', 'r', 's', 'sorry', 't', 'thank you', 'u', 'v', 'w', 'x', 'y', 'yes', 'z']
# test_letters = ['my']
# letters removed = o, 
letter_count = len(test_letters)
labels = []

collection_folder = "./A2E/CameraTracking/ValidationData"
if os.path.exists(collection_folder):
     print("Folder Found")
else:
    collection_folder = './ValidationData'

for subdir, dirs, files in os.walk(collection_folder):
     if subdir.split('\\')[-1] in test_letters:
          for file in files:
               labels.append(subdir.split('\\')[-1])


          
          
sequences = []
for letter in test_letters:
     try:
          dir_length = len(os.listdir(os.path.join(collection_folder, letter)))
          dir_length = len(os.listdir(os.path.join(collection_folder, letter)))
          
          dir_length = len(os.listdir(os.path.join(collection_folder, letter)))          
          
          for i in range(0, dir_length):
               sequences.append(np.load(os.path.join(collection_folder, letter, letter + str(i) + ".npy")))
     except:
          dummy = 0
print("All Data is Loaded")



try:
     model = load_model("./Models/256_full_tanh_model.h5")
except:
     model = load_model("./A2E/CameraTracking/Models/256_full_tanh_model.h5")
print("Model is Loaded")



# accuracies, num_per_label = [], []
predictions = []
for (label, sequence) in zip(labels, sequences):
     # accuracy_count, label_count = 0, 0
     
     prediction = model.predict(np.expand_dims(sequence, axis=0), verbose=0)
     char_index = np.argmax(prediction)
     confidence = round(prediction[0,char_index]*100, 1)
     predicted_char = letters[char_index]
     
     print("Actual: {} | Predicted: {} | Confidence: {}%".format(label, predicted_char, confidence))

     predictions.append(predicted_char)
     

print()
unique_labels = sorted(list(set(labels)))
for label in unique_labels:
    indices = [i for i, x in enumerate(labels) if x == label]
    true_labels_subset = [labels[i] for i in indices]
    predicted_labels_subset = [predictions[i] for i in indices]
    accuracy = accuracy_score(true_labels_subset, predicted_labels_subset)
    print("Letter: {} | Accuracy: {}".format(label, accuracy))

          
accuracy = sum(1 for x,y in zip(labels, predictions) if x == y) / len(labels)
print("\nTotal Accuracy: {}%".format(100*accuracy))