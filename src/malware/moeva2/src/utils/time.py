from functools import wraps
from time import time


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print(f"func:{f.__name__!r} args:[{args!r}, {kw!r}] took: {te - ts:2.4f} sec")
        return result

    return wrap
