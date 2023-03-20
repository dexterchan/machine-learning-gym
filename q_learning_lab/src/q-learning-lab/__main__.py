"""Console script for q_learning_lab."""
import argparse
import sys
from q_learning_lab import execute_lab


def main():
    """Console script for q_learning_lab."""
    parser = argparse.ArgumentParser()
    parser.add_argument('_', nargs='*')
    args = parser.parse_args()

    print("Arguments: " + str(args._))
    print("Replace this message by putting your code into "
          "q_learning_lab.cli.main")
    return 0


if __name__ == "__main__":
    sys.exit(execute_lab())  # pragma: no cover
