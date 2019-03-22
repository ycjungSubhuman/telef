#!/bin/bash

xhost +local:root; \
    nvidia-docker run -it --rm \
    -e DISPLAY=$DISPLAY \
    -e CUDACXX=/opt/cuda/bin/nvcc \
    --privileged \
    -u user \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v $PWD/../:/home/user/src:rw \
    -v /dev/bus/usb:/dev/bus/usb \
    -v /home/$USER/.CLionDocker:/home/user/.CLion2018.1:rw \
    local/telef-build $@
