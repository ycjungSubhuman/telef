#!/bin/bash

# Run face detection server
cd lib/face-alignment/docker && bash build-image.sh &&
    nvidia-docker run --name face-server4 -it \
    -v $PWD/../:/workspace telef-alignment \
    /bin/bash -c "bash build_massges.sh && cd examples/ && python3 detect_server.py"
