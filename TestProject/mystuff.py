from collections import OrderedDict
import numpy as np

__author__ = 'ghash'


class SortedKeysDict(OrderedDict):

    def __init__(self, dictionary):
        """
        Initializes a sorted dictionary, sorting is done by a key value
        :param dictionary: a normal unsorted dictionary or another sorted one
        """
        super(SortedKeysDict, self).__init__(sorted(dictionary.items()))

    # Reading and writing to a file
    # noinspection PyMethodMayBeStatic
    def scale_matrix(self, matrix_file, scale):
        x = np.loadtxt(matrix_file)
        return scale*x





