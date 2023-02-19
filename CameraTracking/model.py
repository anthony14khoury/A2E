import numpy as np
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping


def gathering_data(letters, samples_length, label_map):
    sequences, labels = [], []
    
    for letter in letters:
        
        for i in range(0, samples_length):

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

    print("\t Compiling and Fitting Model")
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, min_delta=0.001, mode='auto')
    history = model.fit(X_train, y_train, epochs=100, verbose=1, validation_data=(X_test, y_test))

    print("Saving Model")
    model.save('abcefjnothing2.h5')
    
    return history


if __name__ == "__main__":
    
    letters = np.array(['a', 'b', 'c', 'e', 'f', 'g', 'h' 'i', 'j', 'k', 'l', 'n', 'nothing'])
    label_map = {label:letters for letters, label in enumerate(letters)}
    samples_length = 20
    
    print("\t Gathering data to input into model")
    sequences, labels = gathering_data(letters, samples_length, label_map)
    
    X = np.array(sequences)
    y = to_categorical(labels).astype(int)
    
    print("\t Creating and Saving the ML Model:")
    history = compute_model(X, y, letters)
    
    print("Plot Learning Curves")
    
    
    



# epochs = 50
# num_classes = 3

# for activation in [None, 'sigmoid', 'tanh', 'relu']:
#     model = Sequential()
#     model.add(Dense(512, activation=activation, input_shape=(30,398)))
#     model.add(Dense(num_classes, activation='softmax'))
#     model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
#     history = model.fit(X_train, y_train, epochs=epochs, verbose=1, validation_data=(X_test, y_test))
    
#     plt.plot(history.history['val_acc'])
