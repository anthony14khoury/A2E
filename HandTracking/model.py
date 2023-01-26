# from sklearn.model_selection import train_test_split
# from tensorflow.keras.utils import to_categorical
import tensorflow
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

LETTERS = np.array(['a', 'j', 'nothing', 'z'])
label_map = {label:LETTERS for LETTERS, label in enumerate(LETTERS)}
sequence_length = 30
samples_length = 10

sequences, labels = [], []
# Loop through actions
for letter in LETTERS:
    for i in range(0,samples_length):
        # Grab all 30 frames and append them to window
        res = np.load(os.path.join("DataCollection", letter, letter+str(i)+".npy"))
        sequences.append(res)
        labels.append(label_map[letter])
        print(res.shape)

X = np.array(sequences)
print(X.shape)
y = to_categorical(labels).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)
log_dir = os.path.join('Logs')
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,398)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(LETTERS.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.fit(X_train, y_train, epochs=500)

model.summary()
res = model.predict(X_test)
model.save('action.h5')
