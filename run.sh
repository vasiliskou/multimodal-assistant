#!/bin/bash

# Optional: build the Docker image
docker build -t multimodal .

# Run the container with port forwarding and .env file (if needed)
docker run -it --rm -p 7860:7860 --env-file .env multimodal