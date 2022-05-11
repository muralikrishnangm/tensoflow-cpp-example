cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(example)

if ("${ARCH}" STREQUAL "CUDA")
  enable_language(CUDA)
endif()

# execute_process(COMMAND "python" "-c" "import tensorflow as tf; import sys; sys.stdout.write(tf.sysconfig.get_lib() + '/')" OUTPUT_VARIABLE TF_LIB_DIR)
# find_library(TF_LIB REQUIRED 
#     NAMES tensorflow_framework 
#     PATHS ${TF_LIB_DIR}
#     NO_DEFAULT_PATH)

# # find_package(TensorFlow REQUIRED tensorflow)
# find_library(TENSORFLOW_LIB libtensorflow REQUIRED HINT ${TF_LIBRARY_DIR})
# link_directories(${TF_LIBRARY_DIR})
# include_directories(${TF_INCLUDE_DIR})
# # add_compile_definitions(${TensorFlow_CFLAGS_OTHER})

if ("${ARCH}" STREQUAL "CUDA")
  set(MAIN_SRC example_tensorflowNN_gpu.cpp)
else()
  set(MAIN_SRC example_tensorflowNN_cpu.cpp)
endif()

add_executable(example ${MAIN_SRC})
# target_link_libraries(example ${TF_LIB})
target_link_libraries(example ${TFFLAGS})
# set_property(TARGET example PROPERTY CXX_STANDARD 14)

if ("${ARCH}" STREQUAL "CUDA")
  enable_language(CUDA)
  set_source_files_properties(${MAIN_SRC} PROPERTIES LANGUAGE CUDA)
endif()
