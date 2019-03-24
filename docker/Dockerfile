FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

RUN export DEBIAN_FRONTEND=noninteractive && \
    export DEBCONF_NONITERACTIVE_SEEN=true && \
    echo "tzdata tzdata/Areas select Asia" > /tmp/tzdata-selection && \
    echo "tzdata tzdata/Zones/Asia select 'Korea (South)'" \
        >> /tmp/tzdata-selection && \
    debconf-set-selections /tmp/tzdata-selection && \
    apt-get update && apt-get install -y tzdata

RUN apt-get update && apt-get install -y \
    git g++ clang cmake \
    python libusb-1.0.0-dev libturbojpeg0-dev libglfw3-dev \
    libxmu-dev libxi-dev libudev-dev freeglut3-dev \
    libceres-dev libeigen3-dev libvtk7-dev libboost1.65-all-dev \
    libflann-dev doxygen ninja-build graphviz libopenni2-dev libopencv-dev \
    autoconf automake libtool curl make unzip

COPY install_dependencies.sh /workspace/install_dependencies.sh
RUN bash /workspace/install_dependencies.sh

WORKDIR /workspace
