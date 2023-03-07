import numpy as np
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras import optimizers
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from parameters import Params
import datetime



def gathering_data(folder, letters, label_map):
    sequences, labels = [], []
    
    for letter in letters:
        
        dir_length = len(os.listdir(os.path.join(folder, letter)))
        
        for i in range(0, dir_length):

            # Grab all 30 frames and append them to window
            res = np.load(os.path.join(folder, letter, letter + str(i) + ".npy"))
            sequences.append(res)
            labels.append(label_map[letter])
            print(os.path.join(folder, letter, letter + str(i) + ".npy"))
    
    return sequences, labels


params = Params()
    
letters = params.LETTERS
label_map = {label:letters for letters, label in enumerate(letters)}

folder = "DataCollection"

# Model File
model_file = "motion_model1.h5"

print("\t Gathering data to input into model")
sequences, labels = gathering_data(folder, letters, label_map)

X = np.array(sequences)
y = to_categorical(labels).astype(int)

  

    
print ("\t Train Test Split")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)

print("\t Defining Model")

activation = 'tanh'
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation=activation, input_shape=(30,126)))
model.add(LSTM(128, return_sequences=True, activation=activation))
model.add(LSTM(64, return_sequences=False, activation=activation))
model.add(Dense(64, activation=activation))
model.add(Dense(32, activation=activation))
model.add(Dense(letters.shape[0], activation='softmax'))

earlystop_callback = EarlyStopping(monitor='categorical_accuracy', min_delta=0.001, patience=12, verbose=1, restore_best_weights=True)

checkpoint = ModelCheckpoint(filepath=model_file, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
optimizer = optimizers.Adam(learning_rate=0.01)
history = model.fit(X_train, y_train, epochs=500, verbose=1, validation_data=(X_test, y_test), callbacks=[tensorboard_callback, checkpoint])

print("Model is Done")

# model.save('static_big_model.h5')