"""Console script for q_learning_lab."""
import argparse
import sys

import json

def execute_lab():
    """Console script for q_learning_lab."""
    parser = argparse.ArgumentParser()
    #Add parser "config" file name here
    parser.add_argument("-c", "--config", default="scripts/config/cart_pole_v1.json" )
    parser.add_argument("-b", "--lab-name", help="lab name", default="CartPole-v1")
    args = parser.parse_args()
    #Load the json file into config dict
    with open(args.config) as f:
        config = json.load(f)
    #Call the execute_lab function

    from .port.lab_run import execute_lab_training
    execute_lab_training(lab_name=args.lab_name, lab_config=config, is_verbose=False)

    return 0


if __name__ == "__main__":
    sys.exit(execute_lab())  # pragma: no cover
