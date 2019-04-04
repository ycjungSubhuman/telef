cmake_minimum_required(VERSION 2.8.3 FATAL_ERROR)

option (BUILD_TESTS "Build All Tests" ON)

project (telef LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS_RELEASE "-O2 ${OpenMP_CXX_FLAGS} -march=native -DUSE_SSE4_INSTRUCTIONS=ON")
set(CUDA_GENCODES
  "-gencode arch=compute_52,code=sm_52 -gencode arch=compute_61,code=sm_61 "
  )
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -rdc=true ${CUDA_GENCODES}")
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G -DDEBUG")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3" )
set(CMAKE_CUDA_HOST_COMPILER gcc-7)

find_package(PCL 1.8 REQUIRED COMPONENTS common io visualization filters kdtree registration)
find_package(Eigen 3.3.4 REQUIRED)
find_package(Boost 1.58 COMPONENTS program_options system filesystem REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core objdetect highgui imgproc flann)
find_package(Ceres REQUIRED)
find_package(VTK REQUIRED)
find_package(dlib REQUIRED)

add_subdirectory(messages)
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

add_library(telef
  include/align/nonrigid_pipe.h
  include/align/rigid_pipe.h
  include/cloud/cloud_pipe.h
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
  include/io/obj.h
  include/io/fakeframe.h
  include/util/fake_frame_record_device.h
  include/io/matrix.h
  include/io/landmark.h
  include/io/dataset/face_warehouse.h
  include/util/convert_arr.h
  include/face/model_cudahelper.h
  include/face/raw_model.h
  include/util/transform.h
  include/feature/feature_detect_frontend.h
  include/io/merger/device_input_merger.h
  include/feature/feature_detect_pipe.h
  include/util/eigen_file_io.h
  include/feature/face.h
  include/util/pcl_cv.h
  include/vis/fitting_visualizer.h
  include/util/po_util.h
  include/face/feeder.h

  src/cloud/cloud_pipe.cpp
  src/align/rigid_pipe.cpp
  src/io/align/align_frontend.cpp
  src/io/ply/meshio.cpp
  src/io/bmp.cpp
  src/io/grabber.cpp
  src/mesh/color_projection_pipe.cpp
  src/align/nonrigid_pipe.cpp
  src/mesh/colormapping.cpp
  src/mesh/mesh.cpp
  src/io/devicecloud.cpp 
  src/io/png.cpp 
  src/io/obj.cpp
  src/io/frontend.cpp 
  src/io/fakeframe.cpp 
  src/io/dataset/face_warehouse.cpp
  src/util/UvPointMapping.cpp 
  src/util/fake_frame_record_device.cpp
  src/io/landmark.cpp
  src/face/model_cudahelper.cpp
  src/face/model.cpp
  src/face/deformation_model.cpp
  src/util/transform.cpp
  src/feature/feature_detect_pipe.cpp
  src/util/transform.cpp
  src/align/lmk_scan_rigid_pipe.cpp
  src/vis/fitting_visualizer.cpp
  src/util/po_util.cpp
  src/face/feeder.cpp)

target_include_directories(telef
  PUBLIC ${OpenCV_INCLUDE_DIRS}
  PUBLIC ${PCL_INCLUDE_DIRS}
  PUBLIC ${Eigen_INCLUDE_DIRS}
  PUBLIC ${Boost_INCLUDE_DIRS}
  PUBLIC ${CERES_INCLUDE_DIRS}
  PUBLIC ${CMAKE_SOURCE_DIR}/include/
  PUBLIC ${CMAKE_SOURCE_DIR}/include/face
  PUBLIC /usr/local/include
  PUBLIC ${Protobuf_INCLUDE_DIRS}
  PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  PUBLIC ${CMAKE_CURRENT_BINARY_DIR} # TODO: Check if needed for protobuf messages
  )

include_directories(${CMAKE_CURRENT_BINARY_DIR})

target_link_libraries(telef
  cutelef
  ${CERES_LIBRARIES}
  opencv_core
  opencv_objdetect
  opencv_imgproc
  opencv_highgui
  opencv_flann
  opencv_ml
  opencv_video
  dlib
  -lcudnn
  -lcusolver
  -lcurand
  ${Protobuf_LIBRARIES}
  messages
  -lglog
  ${PCL_COMMON_LIBRARIES}
  ${PCL_FILTERS_LIBRARIES}
  ${PCL_IO_LIBRARIES}
  ${PCL_KDTREE_LIBRARIES}
  ${PCL_VISUALIZATION_LIBRARIES}
  ${Boost_LIBRARIES} 
  ${VTK_LIBRARIES}
  stdc++fs
  -lGLEW
  -lGL
  glfw
  )


add_library(cutelef
  include/util/cu_quaternion.h
  include/face/cu_model_kernel.h
  include/util/cudautil.h
  include/align/cu_loss.h
  include/util/transform.h
  src/util/cu_quaternion.cu
  src/face/cu_model_kernel.cu
  src/align/cu_loss.cu
  include/util/cu_array.h
  src/util/transform.cpp)

target_include_directories(cutelef
  PUBLIC ${CUDA_INCLUDE_DIRS}
  PUBLIC ${CMAKE_SOURCE_DIR}/include/
  )

target_link_libraries(cutelef -lcublas)
set_target_properties(cutelef PROPERTIES CUDA_SEPARABLE_COMPILATION ON)


add_executable(KinectConnectTest app/tools/KinectConnectTest.cpp)
target_link_libraries(KinectConnectTest telef)

add_executable(PcaTargetFit app/tools/PcaTargetFit.cpp)
target_link_libraries(PcaTargetFit ${Boost_LIBRARIES} telef)

add_executable(MakePcaModel app/tools/MakePcaModel.cpp)
target_link_libraries(MakePcaModel ${Boost_LIBRARIES} telef)

add_executable(MakeFwPcaModel app/tools/MakeFwPcaModel.cpp)
target_link_libraries(MakeFwPcaModel ${Boost_LIBRARIES} telef)

add_executable(RecordFakeFrame app/tools/RecordFakeFrame.cpp)
target_link_libraries(RecordFakeFrame)
target_link_libraries(RecordFakeFrame ${Boost_LIBRARIES} telef)

add_executable(DNNFaceDetector app/byproducts/DNNFaceDetector.cpp)
target_link_libraries(DNNFaceDetector ${Boost_LIBRARIES} telef)

add_executable(FaceTrackerClient app/byproducts/FaceTrackerClient.cpp)
target_link_libraries(FaceTrackerClient ${Boost_LIBRARIES} telef)

if (BUILD_TESTS)
  function(myaddtest testname testpath)
    add_executable(${testname} ${testpath})
    target_link_libraries(${testname} gtest gtest_main gmock ${CBLAS_LIBRARIES})
    target_link_libraries(${testname} telef -lm)
    target_link_libraries(${testname}
      stdc++fs
      opencv_core
      opencv_objdetect
      opencv_imgproc
      opencv_highgui
      opencv_flann
      opencv_ml)
    target_include_directories(${testname}
      PRIVATE ${CMAKE_SOURCE_DIR}/include
      PRIVATE lib/googletest/googletest/include
      PRIVATE lib/googletest/googlemock/include
      PRIVATE ${CBLAS_INCLUDE_DIRS}
      PRIVATE ${EIGEN_INCLUDE_DIRS}
      )
    add_test(NAME ${testname} COMMAND ${testname})
  endfunction(myaddtest)

  add_subdirectory(lib/googletest)
  enable_testing()

  myaddtest(test_cu_quaternion tests/util/test_cu_quaternion.cu)
endif()
