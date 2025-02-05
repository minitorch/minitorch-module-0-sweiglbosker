"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul

def mul(x: float, y: float) -> float:
    return x * y

# - id
def id(x: float) -> float:
    return x

# - add
def add(x: float, y: float) -> float:
    return x + y

# - neg
def neg(x: float) -> float:
    return -x

# - lt
def lt(x: float, y: float) -> bool:
    return (x < y)

# - eq
def eq(x: float, y: float) -> bool:
    return (x == y)

# - max
def max(x: float, y: float) -> float:
    return x if (x > y) else y

# - is_close
def is_close(x: float, y: float) -> bool:
    return abs(x - y) < 1e-2

# - sigmoid
def sigmoid(x: float) -> float:
    if x >= 0:
        return (1/(1 + math.exp(-x)))
    else:
        return (math.exp(x)/(1+math.exp(x)))

# - relu
def relu(x: float) -> float:
    return max(0, x)

# - log
def log(x: float) -> float:
    return math.log(x)

# - exp
def exp(x: float) -> float:
    return math.exp(x)

# - log_back
def log_back(x: float, y: float) -> float:
    return (1/x) * y

# - inv
def inv(x: float) -> float:
    return x ** -1

# - inv_back
def inv_back(x: float, y: float) -> float:
    return (-x ** -2) * y

# - relu_back
def relu_back(x: float, y: float) -> float:
    return 0.0 if x <= 0.0 else y


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists

def map(fn: Callable[[float], float], iterable: Iterable[float]) -> Iterable[float]:
    '''
    Higher-order function that applies a given function to each element of an iterable.
    '''
    return [fn(x) for x in iterable]

def zipWith(fn: Callable[[float, float], float], a: Iterable[float], b: Iterable[float]) -> Iterable[float]:
    '''
    Higher-order function that combines elements from two iterables with the given function
    '''
    return [fn(*x) for x in zip(a, b)]

def reduce(fn: Callable[[float, float], float], lst: Iterable[float], init: float) -> float:
    '''
    Higher-order function that reduces an iterable to a single value using a given function
    '''
    r = init
    for num in lst:
        r = fn(r, num)

    return r

def negList(lst: Iterable[float]) -> Iterable[float]:
    return map(neg, lst)

def addLists(a: Iterable[float], b: Iterable[float]) -> Iterable[float]:
    return zipWith(add, a, b)

def sum(lst: Iterable[float]) -> float:
    return reduce(add, lst, 0.0)

def prod(lst: Iterable[float]) -> float:
    return reduce(mul, lst, 1.0)

# TODO: Implement for Task 0.3.
