cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

project (telef)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(PCL 1.8 REQUIRED COMPONENTS common io visualization filters)
find_package(Eigen 3.3.4 REQUIRED)

link_directories(${PCL_LIBRARY_DIRS})

add_definitions(${PCL_DEFINITIONS})

add_library(telef

        src/io/fetcher.cpp
# for ide support
        include/io/device.h
        include/io/fetcher.h
        include/feature/feature_detector.h
        src/io/device.cpp src/feature/feature_detector.cpp)

target_link_libraries(telef
        ${PCL_COMMON_LIBRARIES}
        ${PCL_IO_LIBRARIES}
        )

include_directories(${PCL_INCLUDE_DIRS})
include_directories(${Eigen_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/include)

add_executable(KinectToPLY app/KinectToPLY.cpp)
target_link_libraries(KinectToPLY
        ${PCL_COMMON_LIBRARIES}
        ${PCL_IO_LIBRARIES} ${PCL_FILTERS_LIBRARIES}
        telef
        )

