from keras.callbacks import TensorBoard, ModelCheckpoint
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical # for windows
# from tensorflow.keras.utils import to_categorical # for linux
from keras.layers import LSTM, Dense
from keras.models import Sequential
from parameters import Params
import numpy as np
import os

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
model_name = "motion_model1.h5"




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



# Save Best Epoch for the Model
checkpoint = ModelCheckpoint(filepath=model_name, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

# Save for Tensorboard
log_dir = "logs/fit/" + model_name
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Compiler Customization
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# Fitting the Model
model.fit(X_train, y_train, epochs=3000, verbose=1, validation_data=(X_test, y_test), callbacks=[tensorboard_callback, checkpoint])



print("Model is Done")