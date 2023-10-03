#!/bin/bash

video_files=($(ls /dev/video*));

docker_command="docker run -it --rm --gpus all -v /tmp/.X11-unix:/tmp/.X11-unix -v $(pwd)/config.yaml:/AI-Collab/simulator/config.yaml:ro -e DISPLAY=$DISPLAY --network host"

for value in "${video_files[@]}"
do
	docker_command="$docker_command --device $value"
done

docker_command="$docker_command julian700/simulator ./start_sim.sh $@"

echo "$docker_command"

eval "$docker_command"
