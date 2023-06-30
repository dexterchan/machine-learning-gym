#!/bin/bash
ARCH=$(uname -m)
VERSION=0.1.0

SUFFIX=${VERSION}-bullseye-$ARCH
#Docker build with argument ARCH
echo Building with $ARCH

docker build -t pigpiggcp/q_learning:$SUFFIX -f scripts/container/Dockerfile --build-arg ARCH=${ARCH} . --target dev
#docker tag pigpiggcp/q_learning:$SUFFIX registry.example.com/q_learning:$SUFFIX