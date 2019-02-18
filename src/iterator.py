import tensorflow as tf
import numpy as np

#Convenient class for iterating through train set randomly
class DatasetIterator:
    def __init__(self, x, y, batch_size):
        assert len(x) == len(y)
        self.x = x
        self.y = y
        self.b_sz = batch_size
        self.b_pt = 0
        self.d_sz = len(x)
        self.idx = None
        self.randomize()

    # randomize indexs in dataset
    def randomize(self):
        self.idx = np.random.permutation(self.d_sz)
        self.b_pt = 0

    # get the next batch of the dataset
    def get_next_batch(self):
        start = self.b_pt
        end = self.b_pt + self.b_sz
        idx = self.idx[start:end]
        x = self.x[idx]
        y = self.y[idx]

        self.b_pt += self.b_sz
        if self.b_pt >= self.d_sz:
            self.randomize()

        return x, y
