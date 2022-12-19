
from .MultiProxy_step1 import MultiProxy
from .MultiProxy_step2 import MultiProxy_step2
from .Test import test
from .Movie import Movie as movie
from .real_world import Real as real

def get_dataset(name):
    return {
        "multi1": MultiProxy,
        "multi2": MultiProxy_step2,
        "test": test,
        "movie":movie,
        "real":real
    }[name]
