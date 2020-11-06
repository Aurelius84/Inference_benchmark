cmake_minimum_required(VERSION 3.3 FATAL_ERROR)


find_package(TensorflowCC REQUIRED)
add_executable(yolov3_exe yolov3.cc)
target_link_libraries(yolov3_exe TensorflowCC::TensorflowCC)

# link cuda if it is available
find_package(CUDA)
if(CUDA_FOUND)
  target_link_libraries(yolov3_exe ${CUDA_LIBRARIES})
endif()
