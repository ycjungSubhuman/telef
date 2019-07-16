export DISPLAY=:0

docker build -t telef:kinect1 --build-arg use_kinect_1=1 . && 
    xhost +local:root; \
    nvidia-docker run -it \
    -v $PWD/../:/workspace \
    --privileged \
    -v /dev/bus/usb:/dev/bus/usb \
    -v /data:/data \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    telef:kinect1 /bin/sh -c 'cd build && exec /bin/bash'
