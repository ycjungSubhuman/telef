cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project (messages LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_VERBOSE_MAKEFILE OFF)

INCLUDE(FindProtobuf)
FIND_PACKAGE(Protobuf 3.0.0 REQUIRED)
include_directories(${Protobuf_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

PROTOBUF_GENERATE_CPP(PROTO_SRCS PROTO_HDRS messages.proto)
PROTOBUF_GENERATE_PYTHON(PROTO_PY messages.proto)

ADD_LIBRARY(messages ${PROTO_HDRS} ${PROTO_SRCS})
target_include_directories(messages PUBLIC ${CMAKE_CURRENT_BINARY_DIR})

add_custom_target(generated_code DEPENDS ${PROTO_SRCS} ${PROTO_HDRS})
add_dependencies(messages generated_code)
target_link_libraries(messages ${Protobuf_LIBRARIES})

# Print path to generated files
message ("PROTO_SRCS = ${PROTO_SRCS}")
message ("PROTO_HDRS = ${PROTO_HDRS}")
message ("PROTO_PY = ${PROTO_PY}")
