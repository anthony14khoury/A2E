import pytest
import numpy as np
import os

labels = ['a', 'b', 'c', 'nothing']

def test_valid_numpy_file():
    for sign in labels:
        target_folder = os.path.join(os.path.join('DataCollection'), sign)
        for i in range(20):
            current_sample_path = target_folder + "/" + sign + str(i) + ".npy"
            assert os.path.exists(current_sample_path)
            assert os.path.getsize(current_sample_path) > 0
            current_sample = np.load(current_sample_path)
            assert current_sample.shape == (30, 398)
            assert isinstance(current_sample[0][0], np.float64) or isinstance(current_sample[0][0], np.int64)
