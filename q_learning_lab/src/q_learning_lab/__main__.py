"""Console script for q_learning_lab."""
import argparse
import sys

import json
from .utility.logging import get_logger
from datetime import datetime
import string
import random

logger = get_logger(__name__)


def execute_lab():
    """Console script for q_learning_lab."""
    parser = argparse.ArgumentParser()
    # Add parser "config" file name here
    parser.add_argument("-c", "--config", default="scripts/config/intraday_config.json")
    parser.add_argument("-b", "--lab-name", help="lab name", default="intraday-market-v0")
    parser.add_argument("-n",  "--force-new", help="force new run", default=False, type=bool)
    parser.add_argument("-i", "--run-id", help="runid, uniquely define run setup. if not set, a random string of character + digit of 8 in length will be provided", type=str)
    args = parser.parse_args()
    # Load the json file into config dict
    with open(args.config) as f:
        config = json.load(f)
    # Call the execute_lab function

    force_new:bool = args.force_new

    run_id:string = args.run_id if args.run_id is not None else "".join(random.choices(string.ascii_lowercase+string.digits), k=8)

    from .port.lab_run import execute_lab_training
    logger.info(f"Training with run id {run_id} start")
    try:
        execute_lab_training(lab_name=args.lab_name, lab_config=config, is_verbose=False, force_new=force_new, run_id=run_id)
    finally:
        logger.info(f"Training with run id {run_id} finished")
    return 0


if __name__ == "__main__":
    sys.exit(execute_lab())  # pragma: no cover
