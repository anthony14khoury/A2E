import tensorflow
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import SGD

print("Imports and Dependences Installed")

LETTERS = np.array(['a', 'b', 'c', 'nothing'])
label_map = {label: LETTERS for LETTERS, label in enumerate(LETTERS)}
sequence_length = 30
samples_length = 20
sequences, labels = [], []

print("Organizing Model Inputs")
for letter in LETTERS:

    for i in range(0, samples_length):
        # Grab all 30 frames and append them to window
        res = np.load(os.path.join("DataCollection", letter, letter + str(i) + ".npy"), allow_pickle=True)
        print(res.shape)
        sequences.append(res)
        labels.append(label_map[letter])
        print(os.path.join("DataCollection", letter, letter + str(i) + ".npy"))

X = np.array(sequences)
y = to_categorical(labels).astype(int)

print("\t Train Test Split")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

# print("\t Defining Model")
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='tanh', input_shape=(30, 398)))
model.add(LSTM(128, return_sequences=True, activation='tanh'))
model.add(LSTM(64, return_sequences=False, activation='tanh'))
model.add(Dense(64, activation='tanh'))
model.add(Dense(32, activation='tanh'))
model.add(Dense(LETTERS.shape[0], activation='softmax'))

# print("\t Compiling and Fitting Model")
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
history = model.fit(X_train, y_train, epochs=650)

model.summary()

print("Saving Model")
model.save('current.h5')
