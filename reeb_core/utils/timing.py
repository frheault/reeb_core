import time
import logging
from contextlib import contextmanager

logger = logging.getLogger("reeb_core")


@contextmanager
def log_timer(label: str):
    """Context manager that logs elapsed time at DEBUG level."""
    t0 = time.perf_counter()
    logger.debug("[Timer] Starting: %s", label)
    yield
    elapsed = time.perf_counter() - t0
    logger.debug("[Timer] Finished: %s in %.4f seconds", label, elapsed)
