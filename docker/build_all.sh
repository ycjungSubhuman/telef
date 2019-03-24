#!/bin/bash
docker build -t telef:kinect1 --build-arg use_kinect_1=1 . &&
    docker build -t telef:kinect2 --build-arg use_kinect_1=0 .

