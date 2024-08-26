#!/bin/bash
current_directory=$(pwd)
mkdir "$current_directory/images_volume"
mkdir "$current_directory/videos_volume"

echo "Start building container for inference, calculating FED and rendering videos"
echo "This may take some time. Representative figure is shown in $current_directory/images_volume"

docker build -t repro:latest -f Dockerfile .
docker run -it -v "$current_directory/images_volume:/app/frames_root/" -v "$current_directory/videos_volume:/app/videos_root/" repro:latest

echo "Start building container for FVD. This may take some time and computational resources"
docker build -t repro2:latest -f Dockerfile2 .

echo "Started running FVD"
docker run -it repro2:latest