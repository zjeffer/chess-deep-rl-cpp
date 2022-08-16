cmake_minimum_required(VERSION 3.23 FATAL_ERROR)

######### MAIN PROJECT #########
project(chess-rl VERSION 1.0)

##### QT ######
find_package(QT NAMES Qt6 REQUIRED COMPONENTS Widgets)
find_package(Qt6 REQUIRED COMPONENTS Widgets)
qt_standard_project_setup()
message(STATUS "QT library status:")
message(STATUS "    QT_VERSION: ${QT_VERSION}")

###### OpenCV ######
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
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

####### csv parsing ######
include_directories("${PROJECT_SOURCE_DIR}/vendor/csv")

###### main files ######
file (GLOB PROJECT_SOURCES 
    ${PROJECT_SOURCE_DIR}/src/*.cc
    ${PROJECT_SOURCE_DIR}/src/*.hh
    ${PROJECT_SOURCE_DIR}/src/chess/*.cc
    ${PROJECT_SOURCE_DIR}/src/chess/*.hh
    ${PROJECT_SOURCE_DIR}/src/logger/*.cc
    ${PROJECT_SOURCE_DIR}/src/logger/*.hh
    ${PROJECT_SOURCE_DIR}/src/neuralNetwork/*.cc
    ${PROJECT_SOURCE_DIR}/src/neuralNetwork/*.hh
    ${PROJECT_SOURCE_DIR}/src/ui/*.cc
    ${PROJECT_SOURCE_DIR}/src/ui/*.hh
    ${PROJECT_SOURCE_DIR}/src/ui/*.ui
)

add_executable(${PROJECT_NAME} ${PROJECT_SOURCES})

# link libraries (opencv, torch, g3log, qt)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${TORCH_LIBRARIES} g3log Qt6::Widgets)

set_target_properties(${PROJECT_NAME} PROPERTIES
    WIN32_EXECUTABLE ON
    MACOSX_BUNDLE ON
    CXX_STANDARD 20
)
# set warnings
target_compile_options(${PROJECT_NAME} PRIVATE 
    -Wall -pedantic 
    -pedantic-errors -Wextra -Wcast-align
    -Wchar-subscripts  -Wcomment
    -Wdisabled-optimization
    -Werror -Wfloat-equal  -Wformat  -Wformat=2
    -Wformat-nonliteral -Wformat-security 
    -Wformat-y2k
    -Wimport  -Winit-self  -Winline
    -Winvalid-pch  
    -Wunsafe-loop-optimizations  -Wlong-long
    -Wmissing-field-initializers -Wmissing-format-attribute  
    -Wmissing-noreturn
    -Wpacked -Wparentheses  -Wpointer-arith
    -Wredundant-decls -Wreturn-type
    -Wsequence-point -Wsign-compare  -Wstack-protector
    -Wstrict-aliasing -Wstrict-aliasing=2 -Wswitch -Wtrigraphs  -Wuninitialized
    -Wunknown-pragmas  -Wunreachable-code -Wunused
    -Wunused-function  -Wunused-label  -Wunused-parameter
    -Wunused-value  -Wunused-variable  -Wvariadic-macros
    # -Wshadow
    -Wvolatile-register-var  -Wwrite-strings)

# fix libgomp error
add_library(libgomp SHARED IMPORTED)
set_property(TARGET libgomp PROPERTY IMPORTED_LOCATION "/usr/lib/libgomp.so")
target_link_libraries(${PROJECT_NAME} libgomp)

