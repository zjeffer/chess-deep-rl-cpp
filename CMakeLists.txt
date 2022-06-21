cmake_minimum_required(VERSION 3.23 FATAL_ERROR)

######### MAIN PROJECT #########
project(chess-rl VERSION 1.0.0)

###### OpenCV ######
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

###### LibTorch ######
set(CMAKE_PREFIX_PATH /usr/include/libtorch/share/cmake/)
find_package(Torch REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${OpenCV_LIBS})
include_directories(${TORCH_INCLUDE_DIRS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

###### g3log #########
set(CMAKE_PREFIX_PATH /usr/local/lib/cmake/g3log)
find_package(g3log CONFIG REQUIRED)
message(
"g3log package:
    found: ${G3LOG_FOUND}
    include dir: ${G3LOG_INCLUDE_DIRS}
    libraries: ${G3LOG_LIBRARIES}"
)
include_directories(${G3LOG_INCLUDE_DIRS})

###### main files ######
file (GLOB SOURCE_FILES 
  ${PROJECT_SOURCE_DIR}/src/*.cc
  ${PROJECT_SOURCE_DIR}/src/*.hh
  ${PROJECT_SOURCE_DIR}/src/chess/*.cc
  ${PROJECT_SOURCE_DIR}/src/chess/*.hh
  ${PROJECT_SOURCE_DIR}/src/logger/*.cc
  ${PROJECT_SOURCE_DIR}/src/logger/*.hh
  ${PROJECT_SOURCE_DIR}/src/neuralNetwork/*.hh
)

# create executable
add_executable(${PROJECT_NAME} ${SOURCE_FILES})
# link libraries (opencv, torch, g3log)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} "${TORCH_LIBRARIES}" g3log)

# fix libgomp error
add_library(libgomp SHARED IMPORTED)
set_property(TARGET libgomp PROPERTY IMPORTED_LOCATION "/usr/lib/libgomp.so")
target_link_libraries(${PROJECT_NAME} libgomp)

# set c++ standard
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 20)
