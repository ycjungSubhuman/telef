docker build -t telef:kinect2 --build-arg use_kinect_1=0 . && 
    xhost +local:root; \
    nvidia-docker run -it \
    -v $PWD/../:/workspace \
    --privileged \
    -v /dev/bus/usb:/dev/bus/usb \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    telef:kinect2 /bin/bash
