"""
Collection of the core mathematical operators used throughout the code base.
"""


import math

# ## Task 0.1

# Implementation of a prelude of elementary functions.


def mul(x, y):
    ":math:`f(x, y) = x * y`"
    f = x * y
    return f


def id(x):
    ":math:`f(x) = x`"
    f = 1 * x
    return f


def add(x, y):
    ":math:`f(x, y) = x + y`"
    f = x + y
    return f


def neg(x):
    ":math:`f(x) = -x`"
    f = -1 * x
    return f


def lt(x, y):
    ":math:`f(x) =` 1.0 if x is less than y else 0.0"
    f = x < y
    return int(f)


def eq(x, y):
    ":math:`f(x) =` 1.0 if x is equal to y else 0.0"
    f = (x == y)
    return int(f)


def max(x, y):
    ":math:`f(x) =` x if x is greater than y else y"
    if (x > y):
        return x
    else:
        return y
    

def is_close(x, y):
    ":math:`f(x) = |x - y| < 1e-2` "
    f = math.fabs(x - y) < 1e-2
    return f


def sigmoid(x):
    r"""
    :math:`f(x) =  \frac{1.0}{(1.0 + e^{-x})}`

    (See `<https://en.wikipedia.org/wiki/Sigmoid_function>`_ .)

    Calculate as

    :math:`f(x) =  \frac{1.0}{(1.0 + e^{-x})}` if x >=0 else :math:`\frac{e^x}{(1.0 + e^{x})}`

    for stability.

    Args:
        x (float): input

    Returns:
        float : sigmoid value
    """
    if (x >= 0):
        f = 1.0/(1.0 + math.exp(-x))
    else:
        f = math.exp(x)/(1.0 + math.exp(x))
    return f


def relu(x):
    """
    :math:`f(x) =` x if x is greater than 0, else 0

    (See `<https://en.wikipedia.org/wiki/Rectifier_(neural_networks)>`_ .)

    Args:
        x (float): input

    Returns:
        float : relu value
    """
    if (x > 0):
        return x
    else:
        return 0
    

EPS = 1e-6


def log(x):
    ":math:`f(x) = log(x)`"
    return math.log(x + EPS)


def exp(x):
    ":math:`f(x) = e^{x}`"
    return math.exp(x)


def log_back(x, d):
    r"If :math:`f = log` as above, compute d :math:`d \times f'(x)`"
    f = d * inv(log(x))
    return f


def inv(x):
    ":math:`f(x) = 1/x`"
    f = 1.0/x 
    return f


def inv_back(x, d):
    r"If :math:`f(x) = 1/x` compute d :math:`d \times f'(x)`"
    f = d* inv(1.0/x)
    return f


def relu_back(x, d):
    r"If :math:`f = relu` compute d :math:`d \times f'(x)`"
    
    return d if x > 0 else 0


# ## Task 0.3

# Small library of elementary higher-order functions for practice.


def map(fn):
    """
    Higher-order map.

    .. image:: figs/Ops/maplist.png


    See `<https://en.wikipedia.org/wiki/Map_(higher-order_function)>`_

    Args:
        fn (one-arg function): Function from one value to one value.

    Returns:
        function : A function that takes a list, applies `fn` to each element, and returns a
        new list
    """
    def apply(ls):
        ret = []
        for i in ls:
            ret.append(fn(i))
        return ret
    return apply



def negList(ls):
    "Use :func:`map` and :func:`neg` to negate each element in `ls`"
    return map(neg)(ls)
    


def zipWith(fn):
    """
    Higher-order zipwith (or map2).

    .. image:: figs/Ops/ziplist.png

    See `<https://en.wikipedia.org/wiki/Map_(higher-order_function)>`_

    Args:
        fn (two-arg function): combine two values

    Returns:
        function : takes two equally sized lists `ls1` and `ls2`, produce a new list by
        applying fn(x, y) on each pair of elements.

    """
    def apply(ls1, ls2):
        ret = []
        if len(ls1) == 0:
            return ls2
        elif len(ls2) == 0:
            return ls1
        else:
            for x, y in zip(ls1, ls2):
                ret.append(fn(x, y))

        return ret
    return apply


def addLists(ls1, ls2):
    "Add the elements of `ls1` and `ls2` using :func:`zipWith` and :func:`add`"
    
    return zipWith(add)(ls1, ls2)


def reduce(fn, start):
    r"""
    Higher-order reduce.

    .. image:: figs/Ops/reducelist.png


    Args:
        fn (two-arg function): combine two values
        start (float): start value :math:`x_0`

    Returns:
        function : function that takes a list `ls` of elements
        :math:`x_1 \ldots x_n` and computes the reduction :math:`fn(x_3, fn(x_2,
        fn(x_1, x_0)))`
    """
    def apply(ls):
        if len(ls) == 0:
            return 0
        else:
            val = start
            for l in ls:
                val = fn(val, l)
        return val
    return apply 
    


def sum(ls):
    "Sum up a list using :func:`reduce` and :func:`add`."
    return reduce(add, 0.0)(ls)


def prod(ls):
    "Product of a list using :func:`reduce` and :func:`mul`."
    return reduce(mul, 1.0)(ls)
    
