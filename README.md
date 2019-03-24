# telef

C++ Library for Facial Performance Capture with Kinect 

## Setup

### Dependency

Read `docker/Dockerfile` and install all the dependencies

#### Or use docker image for this project

1. Install nvidia-docker2
1. Run `./run_bash.sh`
1. Now you are in a bash session where you can compile and test this project

## Build

```
mkdir build
cd build
cmake ../
make
```

## Reference

Face model samples
* Tianye Li*, Timo Bolkart*, Michael J. Black, Hao Li, and Javier Romero. 2017. Learning a model of facial shape and expression from 4D scans. ACM Trans. Graph. 36, 6, Article 194 (November 2017), 17 pages. https://doi.org/10.1145/3130800.3130813
