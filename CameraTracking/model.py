import numpy as np
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping, TensorBoard
from parameters import Params
import datetime
import matplotlib.pyplot as plt

def gathering_data(letters, label_map):
    sequences, labels = [], []
    
    for letter in letters:
        dir_length = len(os.listdir(os.path.join("DataCollection", letter)))
        
        for i in range(0, dir_length):

            # Grab all 30 frames and append them to window
            res = np.load(os.path.join("DataCollection", letter, letter + str(i) + ".npy"))
            sequences.append(res)
            labels.append(label_map[letter])
            print(os.path.join("DataCollection", letter, letter + str(i) + ".npy"))
    
    return sequences, labels


def compute_model(X, y, letters):
    
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
    
    earlystop_callback = EarlyStopping(
        monitor='categorical_accuracy', # monitor validation loss
        min_delta=0.001, # minimum change in the monitored quantity to qualify as an improvement
        patience=12, # number of epochs with no improvement after which training will be stopped
        verbose=1, # print message when training stops
        restore_best_weights=True # restore the weights of the best iteration when stopping training
    )

    print("\t Compiling and Fitting Model")
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    history = model.fit(X_train, y_train, epochs=120, verbose=1, validation_data=(X_test, y_test), callbacks=[tensorboard_callback])

    # print("Plot Learning Curves")
    # plt.plot(history.history['categorical_accuracy'])
    # plt.plot(history.history['val_categorical_accuracy'])
    # plt.title('model accuracy')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.savefig('accuracy_history.png')

    # # summarize history for loss
    # plt.clf()
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.savefig('loss_history.png')

    print("Saving Model")
    
    model.save('bigmodel1.h5')
    
    return history


if __name__ == "__main__":
    
    params = Params()
    
    letters = params.LETTERS
    label_map = {label:letters for letters, label in enumerate(letters)}
    
    print("\t Gathering data to input into model")
    sequences, labels = gathering_data(letters, label_map)
    
    X = np.array(sequences)
    y = to_categorical(labels).astype(int)
    
    print("\t Creating and Saving the ML Model:")
    history = compute_model(X, y, letters)
  