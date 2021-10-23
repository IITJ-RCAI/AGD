# Source: https://medium.com/fintechexplained/advanced-python-learn-how-to-profile-python-code-1068055460f9

from cProfile import Profile
import functools
import pstats
import tempfile


def profile(func):
    @functools.wraps(func)
    def wraps(*args, **kwargs):
        file = tempfile.mktemp()
        profiler = Profile()
        ret = profiler.runcall(func, *args, **kwargs)
        profiler.dump_stats(file)
        metrics = pstats.Stats(file)
        metrics.strip_dirs().sort_stats("time").print_stats(100)
        return ret

    return wraps
