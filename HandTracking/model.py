import tensorflow
import numpy as np
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import SGD

print("Imports and Dependences Installed")


LETTERS = np.array(['a', 'b', 'nothing'])
label_map = {label:LETTERS for LETTERS, label in enumerate(LETTERS)}
sequence_length = 30
samples_length = 20
sequences, labels = [], []

print("Organizing Model Inputs")
for letter in LETTERS:
    
    for i in range(0, samples_length):

        # Grab all 30 frames and append them to window
        res = np.load(os.path.join("DataCollection", letter, letter + str(i) + ".npy"))
        sequences.append(res)
        labels.append(label_map[letter])
        print(os.path.join("DataCollection", letter, letter + str(i) + ".npy"))

X = np.array(sequences)
y = to_categorical(labels).astype(int)


print ("\t Train Test Split")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)


# print("\t Defining Model")
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,398)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(LETTERS.shape[0], activation='softmax'))

# print("\t Compiling and Fitting Model")
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.fit(X_train, y_train, epochs=500)

model.summary()

print("Saving Model")
model.save('abnothing.h5')
