from collections import OrderedDict

__author__ = 'ghash'


class SortedKeysDict(OrderedDict):

    def __init__(self, dictionary):
        """
        Initializes a sorted dictionary, sorting is done by a key value
        :param dictionary: a normal unsorted dictionary or another sorted one
        """
        super(SortedKeysDict, self).__init__(sorted(dictionary.items()))
