import numpy as np
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import cv2


#load_dataset function to load the data and resize the images to 50x50
def load_dataset(directory):
    images = []
    labels = []
    for idx, label in enumerate(uniq_labels):
        print(idx, label)
        for file in os.listdir(directory + '/'+label):
            print(file)
            filepath = directory +'/'+ label + "/" + file
            img = cv2.resize(cv2.imread(filepath),(50,50))
            images.append(img)
            labels.append(idx)
    images = np.asarray(images)
    labels = np.asarray(labels)
    return images, labels


#loading_dataset into X_pre and Y_pre
data_dir = os.path.abspath(os.getcwd())+'/Image_Directory'
uniq_labels = sorted(os.listdir(data_dir))
X_pre, Y_pre = load_dataset(data_dir)
print(X_pre.shape, Y_pre.shape)

#spliting dataset into 80% train, 10% validation and 10% test data
X_train, X_test, Y_train, Y_test = train_test_split(X_pre, Y_pre, test_size = 0.2)
X_test, X_eval, Y_test, Y_eval = train_test_split(X_test, Y_test, test_size = 0.5)


# converting Y_tes and Y_train to One hot vectors using to_categorical
# example of one hot => '1' is represented as [0. 1. 0. . . . . 0.]
Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)
Y_eval = to_categorical(Y_eval)
X_train = X_train / 255.
X_test = X_test/ 255.
X_eval = X_eval/ 255.

# building our model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation ='relu', input_shape=(50,50,3)),
    tf.keras.layers.Conv2D(16, (3,3), activation ='relu'),
    tf.keras.layers.Conv2D(16, (3,3), activation ='relu'),
    tf.keras.layers.MaxPool2D((2,2)),
    tf.keras.layers.Conv2D(32, (3,3), activation ='relu'),
    tf.keras.layers.Conv2D(32, (3,3), activation ='relu'),
    tf.keras.layers.Conv2D(32, (3,3), activation ='relu'),
    tf.keras.layers.MaxPool2D((2,2)),
    tf.keras.layers.Conv2D(64, (3,3), activation ='relu'),
    tf.keras.layers.Conv2D(64, (3,3), activation ='relu'),
    tf.keras.layers.Conv2D(64, (3,3), activation ='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(24, activation='softmax')
])

model.summary()

#compiling the model
#default batch size 32
#default learning rate is 0.001
model.compile(optimizer = 'adam',
              loss = 'categorical_crossentropy',
              metrics=['accuracy'])

#start training(fitting) the data
history = model.fit(X_train, Y_train, epochs=20, verbose=1, validation_data=(X_eval, Y_eval))

#testing
model.evaluate(X_test, Y_test)

#save the model
model.save(os.path.abspath(os.getcwd())+'/model.h5')

train_loss = history.history['loss']
train_acc = history.history['accuracy']
val_loss = history.history['val_loss']
val_accuracy = history.history['val_accuracy']
