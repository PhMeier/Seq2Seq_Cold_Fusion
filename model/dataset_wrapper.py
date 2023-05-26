from __future__ import unicode_literals, print_function, division
from torch.utils.data import Dataset
import numpy as np

"""
Module provides a class for parallel data. Useful for batch training. 
"""

class ParallelData(Dataset):
    """
    Convert parallel Data to tensors for batch processing
    """
    def __init__(self, X, y):
        self.data = X
        self.target = y
        self.length = [np.sum(1 - np.equal(x, 0)) for x in X]

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        x_len = self.length[index]
        return x, y, x_len

    def __len__(self):
        return len(self.data)

