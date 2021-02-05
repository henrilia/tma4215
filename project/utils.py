import time
from functools import wraps
import logging
from typing import Callable
import numpy as np

logger = logging.Logger(__name__)


def timeit(function):
    @wraps(function)
    def timed(*args, **kwargs):
        silent = kwargs.get("silent", None)
        if silent is not None:
            del kwargs["silent"]

        ts = time.time()
        result = function(*args, **kwargs)
        te = time.time()
        if not silent:
            logger.warn(
                f"{function.__name__} executed in {(te-ts)*1000} ms{kwargs2str(**kwargs)}"
            )
        return result

    return timed


def kwargs2str(**kwargs):
    out = ""
    for kwarg in kwargs:
        if callable(kwargs[kwarg]):
            val = kwargs[kwarg].__name__
        else:
            val = kwargs[kwarg]
        out += f", {kwarg} = {val}"

    return out


VectorFunction = Callable[[np.ndarray], np.ndarray]

FloatFunction = Callable[[float], float]
