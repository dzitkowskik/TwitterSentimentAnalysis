from collections import Counter

__author__ = 'ghash'


def is_hashad(number):
    hashad = 0
    for letter in str(number):
        hashad += int(letter)
    if number % hashad == 0:
        return True
    else:
        return False


def test_is_hashad():
    assert is_hashad(81) == True
    assert is_hashad(99) == False
    print "is_hashad test OK"


test_is_hashad()

l = ['a', 'b', 'f', 'f', 'b', 'b']


def count_occurrences(arr):
    cnt = Counter()
    for word in arr:
        cnt[word] += 1
    return dict(cnt)


def count_occurrences_test():
    assert count_occurrences(l)['b'] == 3
    print "count_occurrences test OK"


count_occurrences_test()


def factorial(n):
    if n == 2:
        return n
    return n * factorial(n - 1)


def factorial_test():
    assert factorial(4) == 24
    assert factorial(5) == 120
    print "factorial test OK"


factorial_test()

#print(factorial(10000))

d = {'a': 1, 'c': 2, 'b': 3, 'd': 4}

import mystuff

s = mystuff.SortedKeysDict(d)
print s.keys()
print s.items()

if __name__ == "__main__":
    import doctest
    import doctestExample
    doctest.testmod(doctestExample)



