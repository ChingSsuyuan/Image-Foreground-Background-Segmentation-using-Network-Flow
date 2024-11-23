cmake_minimum_required(VERSION 3.10)
project(ImageUploadExample)

set(CMAKE_CXX_STANDARD 17)

# 查找 OpenCV 包
find_package(OpenCV REQUIRED)

# 查找 FLTK 包
find_package(FLTK REQUIRED)

# 包含 OpenCV 和 FLTK 的头文件路径
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${FLTK_INCLUDE_DIR})

# 创建可执行文件
add_executable(${PROJECT_NAME} main.cpp Segmentation.cpp)

# 链接 OpenCV 和 FLTK 库
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${FLTK_LIBRARIES})