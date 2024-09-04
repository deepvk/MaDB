#!/bin/bash

app=$(pwd)

docker run --name beamform -it --rm \
    --net=host --ipc=host \
    --gpus "1" \
    -v ${app}:/app \
    beamform