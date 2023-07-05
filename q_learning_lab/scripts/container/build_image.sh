#!/bin/bash
ARCH=$(uname -m)
VERSION=0.1.1
if [ -z "$PRIVATE_CONDA_CHANNEL" ]; then
PRIVATE_CONDA_CHANNEL=$(cat scripts/.secret/conda_credential)
fi

SUFFIX=${VERSION}-bullseye-$ARCH
#Docker build with argument ARCH
echo Building with $ARCH

docker build -t pigpiggcp/q_learning:$SUFFIX -f scripts/container/Dockerfile --build-arg ARCH=${ARCH} --build-arg PRIVATE_CONDA_CHANNEL=${PRIVATE_CONDA_CHANNEL} . --target dev
#docker tag pigpiggcp/q_learning:$SUFFIX registry.example.com/q_learning:$SUFFIX