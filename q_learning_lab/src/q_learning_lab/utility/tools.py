import time
from functools import wraps
import os
from .logging import get_logger

logger = get_logger(__name__)
def timeit(verbose: bool = False):
    if "TIME_IT_VERBOSE" in os.environ:
        verbose = True

    def decorator(func):
        timelst = [0]*1

        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            try:
                return func(*args, **kwargs)
            finally:

                t_total = time.time() - start
                # timelst.clear()
                timelst[0] = t_total
                if verbose:
                    t_total_ms = int(t_total * 1000)
                    logger.info(
                        f"{func.__class__.__name__} {func.__name__} took {t_total_ms} ms"
                    )
                pass
            pass

        wrapper.execution_time = timelst
        return wrapper

    return decorator