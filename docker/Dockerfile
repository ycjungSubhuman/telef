FROM nvidia/cudagl:9.2-devel-ubuntu18.04

ARG use_kinect_1=0

RUN apt-get update && apt-get install -y \
    libcudnn7=7.5.0.56-1+cuda9.2 \
    libcudnn7-dev=7.5.0.56-1+cuda9.2

# Install tzdata package
# All the lines are doing is to automate interactive configuration
# If you are setting up an environment in your host machine,
# You can just `apt-get install tzdata` instead of these.
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

ENV USE_KINECT_1=$use_kinect_1
COPY install_dependencies.sh /workspace/install_dependencies.sh
RUN bash /workspace/install_dependencies.sh

WORKDIR /workspace
