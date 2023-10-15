#!/bin/bash

bash  /var/task/mount_s3fs.sh 
python3 -m q_learning_lab -b intraday-market-v0 -c /var/task/run/scripts/config/intraday_config_s3fs.json -i local_intraday
