cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
project (KinectFace)

find_package(PCL 1.8 REQUIRED COMPONENTS common io visualization)

link_directories(${PCL_LIBRARY_DIRS})

add_definitions(${PCL_DEFINITIONS})

include_directories(${PCL_INCLUDE_DIRS})
include_directories(include)

add_library(libKinnectFace src/RGBDFrame.cpp src/io/OpenNI2Reader.cpp)

add_executable(KinnectToPLY app/KinnectToPLY.cpp)
target_link_libraries(KinnectToPLY ${PCL_COMMON_LIBRARIES} ${PCL_IO_LIBRARIES} libKinnectFace)
