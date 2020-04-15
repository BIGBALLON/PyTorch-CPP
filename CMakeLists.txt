cmake_minimum_required(VERSION 3.0)
project(predict_demo)

find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)

include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable(classifier prediction.cpp)
target_link_libraries(classifier ${OpenCV_LIBS} ${TORCH_LIBRARIES})
# set C++14 to compile PyTorch
set_property(TARGET classifier PROPERTY CXX_STANDARD 14)
