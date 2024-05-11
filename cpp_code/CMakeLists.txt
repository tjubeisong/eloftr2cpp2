cmake_minimum_required(VERSION 3.0)
project(loftrtest)

set (DCMAKE_PREFIX_PATH ~/libtorch/bin) 
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(Torch_DIR /home/ububtu-2204/libtorch/share/cmake/Torch)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
#find_package (gflags REQUIRED)

if (NOT CMAKE_BUILD_TYPE)
    message(STATUS "No build type selected, default to Release")
    set(CMAKE_BUILD_TYPE "Release")
endif()
include_directories(${OpenCV_INCLUDE_DIRS})



add_executable(loftrtest LoFTRMatching.cpp)
target_link_libraries(loftrtest ${OpenCV_LIBS} ${TORCH_LIBRARIES})

file(COPY
  /home/ububtu-2204/Mycode/Eloftr/LoFTR_pretrained_cpp/models_script/weights/traced_eloftr_model.zip
  DESTINATION ${CMAKE_BINARY_DIR}
)


