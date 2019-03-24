#!/bin/bash

# build/installation script to be used inside the docker image
# Written by Yucheol Jung <ycjung@postech.ac.kr> 
# Last Update : 2019/03/21

# exit on error

# Configuration
## If set to '1', build dependencies with the most recent master branches
BUILD_WITH_MASTER=0
GIT=git
CMAKE=cmake
#BUILD_TYPE_CMAKE=RelWithDebInfo
BUILD_TYPE_CMAKE=Release
ROOT_SOURCE=/workspace/sources
MKDIR="mkdir -p"
MAKE="make -j$(nproc)"
PREFIX="/usr/local"
PREFIX_SYSTEM="/usr"
DIR_LIB="lib"
DIR_INCLUDE="include"
USE_KINECT_1=${USE_KINECT_1:=0}

## Change these variables to use other compiler i.e.) gcc and g++
CC=clang
CXX=clang++

# Preparation
$MKDIR /workspace/sources

function clone_git () {
    URL=$1
    BRANCH=$2
    SIG=$3
    DST=$4

    if [[ ! -d $DST ]]; then
        $GIT clone --recursive $URL $DST
    fi

    cd $DST

    if [[ 0 == $BUILD_WITH_MASTER ]]; then
        $GIT checkout $BRANCH
        $GIT checkout $SIG
    fi

    cd ../
}


set -xe

## protobuf
VER_PROTOBUF=3.7.0
TARGET_PROTOBUF=protobuf-cpp-$VER_PROTOBUF.tar.gz
TAR_PROTOBUF=$ROOT_SOURCE/$TARGET_PROTOBUF
URL_TAR_PROTOBUF=https://github.com/protocolbuffers/protobuf/releases/download/v$VER_PROTOBUF/$TARGET_PROTOBUF
DST_PROTOBUF=$ROOT_SOURCE/protobuf-$VER_PROTOBUF

curl -L $URL_TAR_PROTOBUF > $TAR_PROTOBUF &&
    tar -xvf $TAR_PROTOBUF -C $ROOT_SOURCE && 
    cd $DST_PROTOBUF &&
    ./configure && $MAKE && $MAKE install &&
    ldconfig


## dlib
URL_DLIB=https://github.com/davisking/dlib
BRANCH_DLIB=master
SIG_DLIB=ae406bf
DST_DLIB=$ROOT_SOURCE/dlib
FLAGS_CMAKE_DLIB="\
    -DCMAKE_BUILD_TYPE=$BUILD_TYPE_CMAKE \
    -DCMAKE_INSTALL_PREFIX=$PREFIX"

clone_git $URL_DLIB $BRANCH_DLIB $SIG_DLIB $DST_DLIB

cd $DST_DLIB && $MKDIR build/ && cd build/ &&
    $CMAKE $FLAGS_CMAKE_DLIB ../ &&
    $MAKE && $MAKE install

DIR_DRIVER_OPENNI2=$PREFIX_SYSTEM/$DIR_LIB/OpenNI2/Drivers

if [[ 1 == $USE_KINECT_1 ]]; then
    ## libfreenect driver for OpenNI2
    URL_FREENECT=https://github.com/OpenKinect/libfreenect
    BRANCH_FREENECT=master
    SIG_FREENECT=aca6046
    DST_FREENECT=$ROOT_SOURCE/libfreenect
    FLAGS_CMAKE_FREENECT="\
        -DCMAKE_BUILD_TYPE=$BUILD_TYPE_CMAKE \
        -DCMAKE_INSTALL_PREFIX=$PREFIX \
        -DBUILD_OPENNI2_DRIVER=ON"

    clone_git $URL_FREENECT $BRANCH_FREENECT $SIG_FREENECT $DST_FREENECT

    cd $DST_FREENECT && $MKDIR build/ && cd build/ &&
        $CMAKE $FLAGS_CMAKE_FREENECT ../ &&
        $MAKE
    cd $DST_FREENECT &&
        cp build/lib/OpenNI2-FreenectDriver/* $DIR_DRIVER_OPENNI2/
else
    ## libfreenect2 driver for OpenNI2
    URL_FREENECT2=https://github.com/OpenKinect/libfreenect2
    BRANCH_FREENECT2=master
    SIG_FREENECT2=dfd4eaf
    DST_FREENECT2=$ROOT_SOURCE/libfreenect2
    FLAGS_CMAKE_FREENECT2="\
        -DCMAKE_BUILD_TYPE=$BUILD_TYPE_CMAKE \
        -DCMAKE_INSTALL_PREFIX=$PREFIX \
        -DBUILD_OPENNI2_DRIVER=ON \
        -DENABLE_CUDA=OFF"

    clone_git $URL_FREENECT2 $BRANCH_FREENECT2 $SIG_FREENECT2 $DST_FREENECT2

    cd $DST_FREENECT2 && $MKDIR build/ && cd build/ &&
        $CMAKE $FLAGS_CMAKE_FREENECT2 ../ &&
        $MAKE

    cd $DST_FREENECT2 &&
        cp build/lib/libfreenect2.so* $PREFIX/$DIR_LIB/ &&
        cp build/lib/libfreenect2-openni2.so* $DIR_DRIVER_OPENNI2/
fi


set +xe

## PCL (Experimental)
## We use experimental version of PCL to use a correct OpenNI2Grabber implementation
URL_PCL=https://github.com/PointCloudLibrary/pcl
BRANCH_PCL=master
SIG_PCL=9bb32e5
DST_PCL=$ROOT_SOURCE/pcl
FLAGS_CMAKE_PCL="\
    -DCMAKE_BUILD_TYPE=$BUILD_TYPE_CMAKE \
    -DWITH_CUDA=OFF \
    -DCMAKE_INSTALL_PREFIX=$PREFIX"

clone_git $URL_PCL $BRANCH_PCL $SIG_PCL $DST_PCL

cd $DST_PCL && $MKDIR build && cd build &&
    $CMAKE $FLAGS_CMAKE_PCL ../ && 
    $MAKE &&
    $MAKE install
