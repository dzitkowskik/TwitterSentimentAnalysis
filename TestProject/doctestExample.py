"""
This is the "example" module.

The example module supplies one function, factorial2().  For example,

>>> factorial2(5)
120
"""


def factorial2(n):
    """Return the factorial2 of n, an exact integer >= 0.

    If the result is small enough to fit in an int, return an int.
    Else return a long.

    >>> [factorial2(n) for n in range(6)]
    [1, 1, 2, 6, 24, 120]
    >>> [factorial2(long(n)) for n in range(6)]
    [1, 1, 2, 6, 24, 120]
    >>> factorial2(30)
    265252859812191058636308480000000L
    >>> factorial2(30L)
    265252859812191058636308480000000L
    >>> factorial2(-1)
    Traceback (most recent call last):
        ...
    ValueError: n must be >= 0

    factorial2s of floats are OK, but the float must be an exact integer:
    >>> factorial2(30.1)
    Traceback (most recent call last):
        ...
    ValueError: n must be exact integer
    >>> factorial2(30.0)
    265252859812191058636308480000000L

    It must also not be ridiculously large:
    >>> factorial2(1e100)
    Traceback (most recent call last):
        ...
    OverflowError: n too large
    """

    import math

    if not n >= 0:
        raise ValueError("n must be >= 0")
    if math.floor(n) != n:
        raise ValueError("n must be exact integer")
    if n + 1 == n:  # catch a value like 1e300
        raise OverflowError("n too large")
    result = 1
    factor = 2
    while factor <= n:
        result *= factor
        factor += 1
    return result