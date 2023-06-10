from q_learning_lab.utility.process_runner import ForkProcessRunner
import os
import time
from q_learning_lab.utility.logging import get_logger
import sys

logger = get_logger(__name__)
def test_fork_processor_fork_child_process() -> None:
    """
        Test the fork_run method of ForkProcessRunner class.
        Args:
            None
        Returns:
            None
    """
    def child_process(wait_time: int) -> int:
        logger.info("Child process %s is running and wait %s seconds", os.getpid(), wait_time)
        time.sleep(wait_time)
        return wait_time
        

    fork_process_runner = ForkProcessRunner()
    fork_process_runner.fork_run(child_process, wait_time=1)
    
    pass

def test_for_processor_fork_child_process_failed() -> None:
    """_summary_
        Test the fork_run method of ForkProcessRunner class.
        Args:
            None
        Returns:
            None
    """
    def child_process(wait_time: int) -> int:
        logger.info("Child process %s is running and wait %s seconds", os.getpid(), wait_time)
        time.sleep(wait_time)
        raise RuntimeError("Child process failed")
    
    fork_process_runner = ForkProcessRunner()
    try:
        fork_process_runner.fork_run(child_process, wait_time=1)
        assert False
    except Exception as ex:
        assert isinstance(ex, RuntimeError)
        pass