__author__ = 'ghash'

d = {'a': 1, 'c': 2, 'b': 3, 'd': 4}

import numpy as np
import mystuff


# noinspection PyClassHasNoInit,PyMethodMayBeStatic
class TestClass:
    def test_answer(self):
        s = mystuff.SortedKeysDict(d)
        assert s.keys()[2] == 'c'

    def test_scale(self):
        s = mystuff.SortedKeysDict(d)
        x = s.scale_matrix('exercise/matrix.txt', 2)
        expected = np.array([[2., 4.], [6., 8.]])
        assert (expected == x).all()



