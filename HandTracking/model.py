# from sklearn.model_selection import train_test_split
# from tensorflow.keras.utils import to_categorical
import tensorflow
import numpy

LETTERS =numpy.array(['a', 'nothing'])
label_map = {label:LETTERS for LETTERS, label in enumerate(LETTERS)}