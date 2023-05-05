#!/bin/bash
ARCH=$(uname -m)
VERSION=0.1.0

SUFFIX=${VERSION}-bullseye-$ARCH
#Docker build with argument ARCH
echo Building with $ARCH
slim build --tag pigpiggcp/q_learning:$SUFFIX --dockerfile scripts/container/Dockerfile --cbo-build-arg ARCH=${ARCH} --http-probe=false  --cbo-target dev .
docker tag pigpiggcp/q_learning:$SUFFIX registry.example.com/q_learning:$SUFFIX