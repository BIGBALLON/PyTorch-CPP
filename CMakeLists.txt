cmake_minimum_required(VERSION 2.8)
project(predict_demo)
SET(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-std=c++11 -O3")

find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)

include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable(classifier prediction.cpp)
target_link_libraries(classifier ${OpenCV_LIBS} ${TORCH_LIBRARIES})
