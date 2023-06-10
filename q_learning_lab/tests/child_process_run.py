import os
import time
import random

def child_process(wait_seconds: int):
    print(f"Child process {os.getpid()} is running")
    time.sleep(wait_seconds)
    print(f"Child process {os.getpid()} is finished")
    pass

def parent_process():
    print(f"Parent process {os.getpid()} is running")

    num_of_child_process = 5

    #Fork consecutive num_of_child_process child processes
    for i in range(num_of_child_process):
        child_pid = os.fork()
        if child_pid == 0:
            # running child process
            child_process(random.randint(1, 5))
            break
        else:
            # parent process
            print(f"Parent process {os.getpid()} is waiting for {i} child process {child_pid} to finish")
            os.wait()
            print(f"Parenet found {i} child process {child_pid} is finished")
    print(f"Parent process {os.getpid()} is finished")
    pass

if __name__ == "__main__":
    parent_process()
    pass