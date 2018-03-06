cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
project (KinectFace)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(PCL 1.8 REQUIRED COMPONENTS common io visualization filters)

link_directories(${PCL_LIBRARY_DIRS})

add_definitions(${PCL_DEFINITIONS})

include_directories(${PCL_INCLUDE_DIRS})
include_directories(include)

add_executable(KinectToPLY app/KinectToPLY.cpp)
target_link_libraries(KinectToPLY ${PCL_COMMON_LIBRARIES} ${PCL_IO_LIBRARIES} ${PCL_FILTERS_LIBRARIES})

