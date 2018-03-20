cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

project (telef)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Required because of relocated static libray. We can delete it after we get rid of it.
set(CMAKE_POSITION_INDEPENDENT_CODE OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -no-pie")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O3")

find_package(PCL 1.8 REQUIRED COMPONENTS common io visualization filters)
find_package(Eigen 3.3.4 REQUIRED)
find_package(Boost 1.58 REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core objdetect highgui imgproc flann)
find_package(Qt5 REQUIRED COMPONENTS Test Concurrent)

link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

link_directories(${OpenCV_LIBRARY_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})

include_directories(${PCL_INCLUDE_DIRS})
include_directories(${Eigen_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/include/face)

add_library(telef
        # hdrs
        include/feature/feature_detector.h
        include/io/channel.h
        include/io/device.h
        include/io/pipe.h
        include/cloud/cloud_pipe.h
        include/type.h
        include/util/booststd.h
        include/io/grabber.h
        include/io/merger.h
        include/io/frontend.h
        include/image/image_pipe.h

        # src
        src/cloud/cloud_pipe.cpp
        src/feature/intraface_detector.cpp
        include/io/opencv_metadata_wrapper.h include/util/UvPointMapping.h)
target_link_libraries(telef ${CMAKE_SOURCE_DIR}/lib/libface.a)
target_link_libraries(telef
        ${PCL_COMMON_LIBRARIES}
        ${PCL_FILTERS_LIBRARIES}
        ${PCL_IO_LIBRARIES}
        )
add_library(modelio src/io/modelio.cpp)

add_executable(KinectTest app/KinectTest.cpp)
target_link_libraries(KinectTest
        telef
        ${PCL_IO_LIBRARIES}
        )

add_executable(LandmarkProject app/LandmarkProject.cpp src/io/modelio.cpp)
target_link_libraries(LandmarkProject
        telef
        ${PCL_IO_LIBRARIES}
        ${PCL_VISUALIZATION_LIBRARIES}
        opencv_core
        opencv_objdetect
        opencv_highgui
        opencv_imgproc
        opencv_flann
        )


add_executable(FaceFeatureDetect app/FaceFeatureDetect.cpp src/io/modelio.cpp)
target_link_libraries(FaceFeatureDetect ${CMAKE_SOURCE_DIR}/lib/libface.a)
target_link_libraries(FaceFeatureDetect
    telef
    opencv_core
    opencv_objdetect
    opencv_highgui
    opencv_imgproc
    opencv_flann
    )
