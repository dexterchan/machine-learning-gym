#!/bin/sh
sudo yum install -y jq

SecretString=$(aws secretsmanager --region ap-northeast-1 get-secret-value --secret-id dev/deeplearning/kraken | jq -r ".SecretString")
access_key_id=$(echo $SecretString | jq -r ".access_key_id")
secret_access_key=$(echo $SecretString | jq -r ".secret_access_key")

#ref: https://github.com/s3fs-fuse/s3fs-fuse
docker run -it --rm --privileged \
-e LOG_FILE=/var/task/run/tmplog/info.log \
-e DATA_DIR=/var/task/run/cryptodata_analysis \
-e aws_access_key_id=${access_key_id} \
-e aws_secret_access_key=${secret_access_key} \
-e MNT_POINT=/var/task/run \
-e S3_BUCKET=boar-bot-2023-ap-northeast-1 \
pigpiggcp/q_learning:0.1.1-bullseye-x86_64 \
execute_s3_training.sh