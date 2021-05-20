# telef

C++ Library for Facial Performance Capture with Kinect 

## Setup

Clone this repository using `git clone --recursive <URL>`

If you have already cloned the repository and can't find `lib/face-alignmtne`, issue `git submodule update --init --recursive`.

### Dependency

Read `docker/Dockerfile` and install all the dependencies

#### Or use docker image for this project

1. Install nvidia-docker2
1. Run `./run_bash_kinect2.sh` if you want to use Kinect v2. `./run_bash_kinect1.sh` otherwise.
1. Now you are in a bash session where you can compile and test this project

All the binaries generated in this session may not run outside of the docker image.

To build in the session,

`mkdir build && cd build && cmake ../ && make -j$(nproc)`

#### Demo (Real-time 3D Face Fitting using PyTorch 3DFAN)

1. Download pre-recorded sequence [Google Drive](https://drive.google.com/file/d/1nkaSN5eUxHexwP11FEWXasgs1QmX3mle/view?usp=sharing)
1. Unzip the file and place the folder as `jake` under `build/`
1. Run `1.face_detector_server_demo.sh` in your host machine (not in the Docker container!)
1. Wait for the server to be initialized
1. Run `2.run_tracker.sh` in your host machine (not in the Docker container!)
1. After all exepiments are done, run `3.clean_demo.sh` before relaunching `1.face_detector_server_demo.sh`

If you want to try your Kinect instead of pre-recorded video, you can delete `-F /jake/` line in `2.run_tacker.sh`


#### Recording a video sequence

1. Locate `RecordFaceFrame` in `build/` folder after build
1. Run `./RecordFaceFrame <output-dir-name>`
1. After a while, enter "q" and "Enter" to your terminal emulator
1. Wait while the program finishes the job.

## Known Issues

* Fitting does not work with Kinect v2 (Grabber implementation have bug right now)
* The project is not real-time yet

## Note

This project is obsolete and unmaintained now. This code was written when I was very new to 3D face reconstruction and tracking. Since the C++ code I wrote was not very easy to maintain and my teammates being more familiar with Python, we moved on to another code based on Python. Unfortunately, because of license issues, we cannot share the new code. My suggestion for this codebase is: You can look up some example codes for using PCL grabber and basic fitting methods for 3D face model. But don't spend too much time on making this code run on your computer. I did not tested this code running on different computers other than my labmates', which contained pre-installed dependencies and data files.
