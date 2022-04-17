cmake_minimum_required(VERSION 3.23 FATAL_ERROR)
project(chess-rl VERSION 1.0)

find_package( OpenCV REQUIRED )

message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

set(CMAKE_PREFIX_PATH /usr/include/libtorch/share/cmake/)

find_package(Torch REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${OpenCV_LIBS})
include_directories(${TORCH_INCLUDE_DIRS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")


file (GLOB SOURCE_FILES 
  ${PROJECT_SOURCE_DIR}/src/*.cc
  ${PROJECT_SOURCE_DIR}/src/chess/*.cc
)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} "${TORCH_LIBRARIES}" )
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 20)
