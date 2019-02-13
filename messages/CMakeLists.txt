cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

INCLUDE(FindProtobuf)
FIND_PACKAGE(Protobuf 3.6.1 REQUIRED)
INCLUDE_DIRECTORIES(${PROTOBUF_INCLUDE_DIR})

PROTOBUF_GENERATE_CPP(CPP_PROTO_SRC CPP_PROTO_HEADER messages.proto)
#PROTOBUF_GENERATE_PYTHON(PY_PROTO_SRC PY_PROTO_HEADER messages.proto)

ADD_LIBRARY(messages ${CPP_PROTO_HEADER} ${CPP_PROTO_SRC})
target_include_directories(messages PUBLIC ${CMAKE_CURRENT_BINARY_DIR})

add_custom_target(generated_code DEPENDS ${CPP_PROTO_SRC} ${CPP_PROTO_HEADER})
add_dependencies(messages generated_code)

# Print path to generated files
message ("CPP_PROTO_SRC = ${CPP_PROTO_SRC}")
message ("CPP_PROTO_HEADER = ${CPP_PROTO_HEADER}")
#message ("PYTHON_PROTO_SRC = ${PY_PROTO_SRC}")
#message ("PYTHON_PROTO_HEADER = ${PY_PROTO_HEADER}")