#!/bin/bash
IFS=$'\n\t'

#export S3_ACL=${S3_ACL:-private}

mkdir -p ${MNT_POINT}

if [ -z "${AWS_ROLE}" ]; then
  echo "${aws_access_key_id}:${aws_secret_access_key}" > /etc/passwd-s3fs
  chmod 0400 /etc/passwd-s3fs

  echo 'AWS_ROLE is not set - mounting S3 with credentials from ENV'
  #/usr/bin/s3fs  ${S3_BUCKET} ${MNT_POINT} -d -d -f -o endpoint=${S3_REGION},allow_other,retries=5
  s3fs -o use_cache=/tmp/s3fs $S3_BUCKET $MNT_POINT
  echo "Mounted ${S3_BUCKET} to ${MNT_POINT} via s3fs"
else
  echo 'AWS_ROLE is set - using it to mount S3'
  #/usr/bin/s3fs ${S3_BUCKET} ${MNT_POINT} -d -d -f -o endpoint=${S3_REGION},iam_role=${AWS_ROLE},allow_other,retries=5
  s3fs -o iam_role=${AWS_ROLE} -o use_cache=/tmp/s3fs $S3_BUCKET $MNT_POINT
  echo "Mounted ${S3_BUCKET} to ${MNT_POINT} via s3fs"
fi