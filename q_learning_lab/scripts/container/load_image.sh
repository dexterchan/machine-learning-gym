#!/bin/bash

AWS_ACCOUNT_ID=192592784707
AWS_DEFAULT_REGION=us-west-2
IMAGE_REPO_NAME=q-learning
VERSION=0.1.0
ARCH=aarch64
LOCAL_REGISTRY=registry.example.com

aws ecr get-login-password --region ${AWS_DEFAULT_REGION} | docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_DEFAULT_REGION}.amazonaws.com
docker pull ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_DEFAULT_REGION}.amazonaws.com/$IMAGE_REPO_NAME:${VERSION}-bullseye-$ARCH
docker tag ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_DEFAULT_REGION}.amazonaws.com/$IMAGE_REPO_NAME:${VERSION}-bullseye-$ARCH ${LOCAL_REGISTRY}/$IMAGE_REPO_NAME:${VERSION}-bullseye-$ARCH
docker push ${LOCAL_REGISTRY}/$IMAGE_REPO_NAME:${VERSION}-bullseye-$ARCH