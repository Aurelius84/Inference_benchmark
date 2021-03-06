project(cpp_inference CXX C)
cmake_minimum_required(VERSION 3.16)

option(WITH_MKL "MKL/OpenBlas Support" ON)
option(WITH_GPU  "Support GPU" ON)
option(WITH_STATIC_LIB "with static/shared library" ON)
option(USE_TENSORRT "Compiled with TensorRT" OFF)


macro(safe_set_static_flag)
    foreach(flag_var CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
        if(${flag_var} MATCHES "/MD")
            string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
        endif(${flag_var} MATCHES "/MD")
    endforeach()
endmacro()


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g")
set(CMAKE_STATIC_LIBRARY_PREFIX "")
message("flags" ${CMAKE_CXX_FLAGSd})

if(NOT DEFINED PADDLE_LIB)
    message(FATAL_ERROR "need -DPADDLE_LIB=/path/paddle/lib")
endif()


include_directories("${PADDLE_LIB}")
include_directories("${PADDLE_LIB}/third_party/install/protobuf/include")
include_directories("${PADDLE_LIB}/third_party/install/glog/include")
include_directories("${PADDLE_LIB}/third_party/install/gflags/include")
include_directories("${PADDLE_LIB}/third_party/install/xxhash/include")
include_directories("${PADDLE_LIB}/third_party/zlib/zlib/include")
include_directories("${PADDLE_LIB}/third_party/install/boost")
include_directories("${PADDLE_LIB}/third_party/install/eigen3")

if(USE_TENSORRT AND WITH_GPU)
    include_directories("${TENSORRT_ROOT}/include")
    link_directories("${TENSORRT_ROOT}/lib")
endif()

link_directories("${PADDLE_LIB}/third_party/install/zlib/lib")
link_directories("${PADDLE_LIB}/third_party/install/glog/lib")
link_directories("${PADDLE_LIB}/third_party/install/gflags/lib")
link_directories("${PADDLE_LIB}/third_party/install/xxhash/lib")
link_directories("${PADDLE_LIB}/paddle/lib")


if(WITH_MKL)
    include_directories("${PADDLE_LIB}/third_party/install/mklml/lib")
    set(MATH_LIB ${PADDLE_LIB}/third_party/install/mklml/lib/libmklml_intel${CMAKE_SHARED_LIBRARY_SUFFIX} 
    ${PADDLE_LIB}/third_party/install/mklml/lib/libiomp5${CMAKE_SHARED_LIBRARY_SUFFIX})
set(MKLDNN_PATH "{PADDLE_LIB}/third_party/install/mkldnn")
if(EXISTS ${MKLDNN_PATH})
    include_directories("${MKLDNN_PATH}/include")
    set(MKLDNN_LIB ${MKLDNN_PATH}/lib/libmkldnn.so.0)
endif()
else()
set(MATH_LIB ${PADDLE_LIB}/third_party/install/openblas/lib/libopenblas${CMAKE_STATIC_LIBRARY_SUFFIX})
endif()

if(WITH_STATIC_LIB)
    set(DEPS ${PADDLE_LIB}/paddle/lib/libpaddle_fluid${CMAKE_STATIC_LIBRARY_SUFFIX})
else()
    set(DEPS ${PADDLE_LIB}/paddle/lib/libpaddle_fluid${CMAKE_SHARED_LIBRARY_SUFFIX})
endif()


set(EXTERNAL_LIB "-lrt -ldl -lpthread")
set(DEPS ${DEPS} ${MATH_LIB} ${MKLDNN_LIB} glog gflags protobuf xxhash ${EXTERNAL_LIB})

if(WITH_GPU)
    if(USE_TENSORRT)
        set(DEPS ${DEPS} ${TENSORRT_ROOT}/lib/libnvinfer${CMAKE_SHARED_LIBRARY_SUFFIX})
        set(DEPS ${DEPS} ${TENSORRT_ROOT}/lib/libnvinfer_plugin${CMAKE_SHARED_LIBRARY_SUFFIX})
    endif()
    set(DEPS ${DEPS} ${CUDA_LIB}/libcudart${CMAKE_SHARED_LIBRARY_SUFFIX})
    set(DEPS ${DEPS} ${CUDNN_LIB}/libcublas${CMAKE_SHARED_LIBRARY_SUFFIX})
    set(DEPS ${DEPS} ${CUDNN_LIB}/libcudnn${CMAKE_SHARED_LIBRARY_SUFFIX})
endif()


add_executable(yolov3_exe yolov3.cc)
target_link_libraries(yolov3_exe ${DEPS})

    

