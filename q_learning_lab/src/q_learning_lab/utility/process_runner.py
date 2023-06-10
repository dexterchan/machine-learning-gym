from __future__ import annotations
import os
from .logging import get_logger
from typing import Any
import sys
logger = get_logger(__name__)



class ForkProcessRunner:
    def __init__(self):
        pass

    def fork_run(self, child_process: callable, *args, **kwargs) -> Any:
        child_pid = os.fork()
        if child_pid == 0:
            # running child process
            logger.info("Child process %s is running", os.getpid())
            try:
                child_process(*args, **kwargs)
            except Exception as ex:
                os._exit(1)
                pass
            os._exit(0)
            pass
        else:
            # parent process
            logger.info("Parent process %s is waiting for child process %s to finish", os.getpid(), child_pid)
            pid, exit_code = os.wait()
            logger.info("Parent process %s is finished", os.getpid())
            if exit_code != 0:
                raise RuntimeError(f"Child process {pid} failed with exit code {exit_code}")
            pass
        pass