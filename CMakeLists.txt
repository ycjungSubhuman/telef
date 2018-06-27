cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

option (BUILD_TESTS "Build All Tests" ON)

project (telef LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_VERBOSE_MAKEFILE OFF)

# Required because of relocated static libray. We can delete it after we get rid of it.
set(CMAKE_POSITION_INDEPENDENT_CODE OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -no-pie")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O2 ${OpenMP_CXX_FLAGS} -march=native")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -rdc=true -gencode arch=compute_61,code=sm_61 -lcublas_device")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3" )

find_package(PCL 1.8 REQUIRED COMPONENTS common io visualization filters kdtree registration)
find_package(Eigen 3.3.4 REQUIRED)
find_package(Boost 1.58 COMPONENTS program_options REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core objdetect highgui imgproc flann)
find_package(Ceres REQUIRED)
find_package(CUDA)
find_package(Qt5 REQUIRED COMPONENTS Test Concurrent)

link_directories(${PCL_LIBRARY_DIRS})

link_directories(${OpenCV_LIBRARY_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${PCL_INCLUDE_DIRS})
include_directories(${Eigen_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${Ceres_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/include/face)

add_library(telef
        include/align/nonrigid_pipe.h
        include/align/rigid.h
        include/align/rigid_pipe.h
        include/cloud/cloud_pipe.h
        include/face/binary_model_file.h
        include/face/FaceAlignment.h
        include/face/XXDescriptor.h
        include/face/model.h
        include/feature/feature_detector.h
        include/io/align/align_frontend.h
        include/io/ply/meshio.h
        include/io/ply/YaPly.h
        include/io/bmp.h
        include/io/channel.h
        include/io/device.h
        include/io/frontend.h
        include/io/grabber.h
        include/io/merger.h
        include/io/opencv_metadata_wrapper.h
        include/io/pipe.h
        include/mesh/color_projection_pipe.h
        include/mesh/colormapping.h
        include/mesh/mesh.h
        include/util/eigen_pcl.h
        include/util/UvPointMapping.h
        include/camera.h
        include/type.h
	include/align/cost_func.h
        include/face/classified_model.h 
        include/face/classify_pipe.h 
        include/face/deformation_model.h
        include/io/devicecloud.h 
        include/io/png.h 
        include/io/fakeframe.h 
        include/util/fake_frame_record_device.h
	include/io/matrix.h
	include/io/landmark.h
	include/util/convert_arr.h
		include/face/model_cudahelper.h

        src/cloud/cloud_pipe.cpp
        src/feature/intraface_detector.cpp
        src/align/rigid_pipe.cpp
        src/io/align/align_frontend.cpp
        src/io/ply/meshio.cpp
        src/io/bmp.cpp
        src/mesh/color_projection_pipe.cpp
        src/align/nonrigid_pipe.cpp
        src/mesh/colormapping.cpp
        src/mesh/mesh.cpp
        src/io/devicecloud.cpp 
        src/io/png.cpp 
        src/io/frontend.cpp 
        src/io/fakeframe.cpp 
        src/util/UvPointMapping.cpp 
        src/util/fake_frame_record_device.cpp
	src/io/landmark.cpp
		src/face/model_cudahelper.cpp
		)

add_library(cutelef STATIC
        include/util/cu_quaternion.h
		include/face/cu_model_kernel.h
		include/util/cudautil.h
		include/align/cu_loss.h

		src/util/cu_quaternion.cu
		src/face/cu_model_kernel.cu
		src/align/cu_loss.cu
		)

set_target_properties(cutelef PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(telef ${CMAKE_SOURCE_DIR}/lib/libface.a)
target_link_libraries(telef
        stdc++fs
		-lcublas_device
    	cutelef
        ${Ceres_LIBRARIES}
        ${PCL_COMMON_LIBRARIES}
        ${PCL_FILTERS_LIBRARIES}
        ${PCL_IO_LIBRARIES}
        ${PCL_VISUALIZATION_LIBRARIES}
        )
add_library(modelio src/io/modelio.cpp)

add_executable(KinectTest app/KinectTest.cpp)
target_link_libraries(KinectTest
		-lcublas_device
        telef
        ${PCL_IO_LIBRARIES}
        opencv_core
        opencv_imgproc
        opencv_highgui
        )

add_executable(LandmarkProject app/LandmarkProject.cpp src/io/modelio.cpp)
target_link_libraries(LandmarkProject ${CMAKE_SOURCE_DIR}/lib/libface.a)
target_link_libraries(LandmarkProject
		-lcublas_device
        stdc++fs
        ${Boost_LIBRARIES}
        telef
        ${PCL_IO_LIBRARIES}
        ${PCL_VISUALIZATION_LIBRARIES}
        opencv_core
        opencv_objdetect
        opencv_imgproc
        opencv_highgui
        opencv_flann
        )


add_executable(FaceFeatureDetect app/FaceFeatureDetect.cpp src/io/modelio.cpp include/io/ply/meshio.h)
target_link_libraries(FaceFeatureDetect ${CMAKE_SOURCE_DIR}/lib/libface.a)
target_link_libraries(FaceFeatureDetect
		-lcublas_device
        telef
        opencv_core
        opencv_objdetect
        opencv_highgui
        opencv_imgproc
        opencv_flann
	-lstdc++
        )

add_executable(PcaTest app/PcaTest.cpp)
target_link_libraries(PcaTest stdc++fs telef ${Boost_LIBRARIES}
		-lcublas_device
        opencv_core
        opencv_ml
        opencv_objdetect
        opencv_highgui
        opencv_imgproc
        opencv_flann
	-lstdc++
        )

add_executable(PcaSample app/PcaSample.cpp)
target_link_libraries(PcaSample stdc++fs telef
		-lcublas_device
        ${Boost_LIBRARIES}
        opencv_core
        opencv_ml
        opencv_objdetect
        opencv_highgui
        opencv_imgproc
        opencv_flann
	-lstdc++
        )

add_executable(RigidAlighnment app/RigidAlighnment.cpp)
target_link_libraries(RigidAlighnment
		-lcublas_device
        stdc++fs
        ${Boost_LIBRARIES}
        telef
        ${PCL_IO_LIBRARIES}
        ${PCL_VISUALIZATION_LIBRARIES}
        opencv_core
        opencv_ml
        opencv_objdetect
        opencv_highgui
        opencv_imgproc
        opencv_flann
	-lstdc++
        )

add_executable(RigidAlignmentTest app/RigidAlignmentTest.cpp src/io/modelio.cpp)
target_link_libraries(RigidAlignmentTest ${CMAKE_SOURCE_DIR}/lib/libface.a)
target_link_libraries(RigidAlignmentTest
		-lcublas_device
        stdc++fs
        telef
        ${Boost_LIBRAREIS}
        ${PCL_IO_LIBRARIES}
        ${PCL_VISUALIZATION_LIBRARIES}
        opencv_core
        opencv_objdetect
        opencv_imgproc
        opencv_highgui
        opencv_flann
	-lstdc++
        )

add_executable(PcaTargetFit app/PcaTargetFit.cpp src/io/modelio.cpp)
target_link_libraries(PcaTargetFit ${CMAKE_SOURCE_DIR}/lib/libface.a)
target_link_libraries(PcaTargetFit
		-lcublas_device
        stdc++fs
        ${Boost_LIBRARIES}
        telef
        ${PCL_IO_LIBRARIES}
        ${PCL_VISUALIZATION_LIBRARIES}
        ${PCL_KDTREE_LIBRARIES}
        opencv_core
        opencv_ml
        opencv_objdetect
        opencv_imgproc
        opencv_highgui
        opencv_flann
        ceres
	-lstdc++
        )

add_executable(PcaFit app/PcaFit.cpp src/io/modelio.cpp)
target_link_libraries(PcaFit ${CMAKE_SOURCE_DIR}/lib/libface.a)
target_link_libraries(PcaFit
		-lcublas_device
        stdc++fs
        ${Boost_LIBRARIES}
        telef
        ${PCL_IO_LIBRARIES}
        ${PCL_VISUALIZATION_LIBRARIES}
        ${PCL_KDTREE_LIBRARIES}
        opencv_core
        opencv_ml
        opencv_objdetect
        opencv_imgproc
        opencv_highgui
        opencv_flann
        ceres
	-lstdc++
        )

add_executable(ConvertPlyUV app/ConvertPlyUV.cpp src/io/modelio.cpp include/face/raw_model.h)
target_link_libraries(ConvertPlyUV ${CMAKE_SOURCE_DIR}/lib/libface.a)
target_link_libraries(ConvertPlyUV
		-lcublas_device
        stdc++fs
        ${Boost_LIBRARIES}
        telef
        ${PCL_IO_LIBRARIES}
        ${PCL_VISUALIZATION_LIBRARIES}
        ${PCL_KDTREE_LIBRARIES}
        opencv_core
        opencv_ml
        opencv_objdetect
        opencv_imgproc
        opencv_highgui
        opencv_flann
	-lstdc++
        )

add_executable(MakePcaModel app/MakePcaModel.cpp src/io/modelio.cpp)
target_link_libraries(MakePcaModel ${CMAKE_SOURCE_DIR}/lib/libface.a)
target_link_libraries(MakePcaModel
		-lcublas_device
        stdc++fs
        ${Boost_LIBRARIES}
        telef
        ${PCL_IO_LIBRARIES}
        ${PCL_VISUALIZATION_LIBRARIES}
        ${PCL_KDTREE_LIBRARIES}
        ${PCL_REGISTRATION_LIBRARIES}
        opencv_core
        opencv_ml
        opencv_objdetect
        opencv_imgproc
        opencv_highgui
        opencv_flann
	-lstdc++
        )

add_executable(PngTest app/PngTest.cpp)
target_link_libraries(PngTest)
target_link_libraries(PngTest
		-lcublas_device
        stdc++fs
        ${Boost_LIBRARIES}
        telef
        ${PCL_IO_LIBRARIES}
	-lstdc++
        )

add_executable(RecordFakeFrame app/RecordFakeFrame.cpp)
target_link_libraries(RecordFakeFrame)
target_link_libraries(RecordFakeFrame
		-lcublas_device
        stdc++fs
        ${Boost_LIBRARIES}
        telef
        ${PCL_IO_LIBRARIES}
        ${PCL_VISUALIZATION_LIBRARIES}
        opencv_core
        opencv_ml
        opencv_objdetect
        opencv_imgproc
        opencv_highgui
        opencv_flann
	-lstdc++
        )

add_executable(FakeDeviceTest app/FakeDeviceTest.cpp src/io/modelio.cpp)
target_link_libraries(FakeDeviceTest ${CMAKE_SOURCE_DIR}/lib/libface.a)
target_link_libraries(FakeDeviceTest
		-lcublas_device
        stdc++fs
        ${Boost_LIBRARIES}
        telef
        ${PCL_IO_LIBRARIES}
        ${PCL_VISUALIZATION_LIBRARIES}
        opencv_core
        opencv_ml
        opencv_objdetect
        opencv_imgproc
        opencv_highgui
        opencv_flann
        )

if (BUILD_TESTS)
	add_subdirectory(lib/googletest)
	enable_testing()
	include_directories(lib/googletest/googletest/include)
	include_directories(tests)
	add_executable(test_cu_quaternion tests/util/test_cu_quaternion.cu tests/testutil/diff.h)
	target_link_libraries(test_cu_quaternion gtest gtest_main -lcublas_device)
	target_link_libraries(test_cu_quaternion cutelef -lm)
	add_test(NAME test_cu_quaternion COMMAND test_cu_quaternion)
endif()
