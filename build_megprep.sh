#!/bin/bash  

IMAGE_NAME=cmrlab/megprep
VERSION="0.0.3"
DOCKERFILE_NAME=megprep.Dockerfile 


if [[ ! -f "$DOCKERFILE_NAME" ]]; then  
    echo "Error: Dockerfile not found at $DOCKERFILE_PATH"  
    exit 1  
fi  


echo "Building Docker image: $IMAGE_NAME using Dockerfile at $DOCKERFILE_PATH..."  
docker build -t "$IMAGE_NAME:$VERSION" -f "$DOCKERFILE_NAME" .


if [[ $? -eq 0 ]]; then  
    echo "Docker image $IMAGE_NAME built successfully."  
else  
    echo "Error: Docker image $IMAGE_NAME failed to build."  
    exit 1  
fi
