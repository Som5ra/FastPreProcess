cmake_minimum_required(VERSION 3.5...3.28)

project(fastpreprocess)


enable_testing()

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS true)

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# Python Interpreter
set(Python3_ROOT_DIR "/home/risksis/anaconda3/envs/fastpreprocess")
find_package(Python3 3.9 EXACT COMPONENTS Interpreter Development REQUIRED)
message("Python_FOUND:${Python3_EXECUTABLE}")




# pybind11
add_subdirectory(pybind11)
include_directories(${pybind11_INCLUDE_DIRS})

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})



pybind11_add_module(fastpreprocess src/preprocess_pybind11.cpp)


target_link_libraries(fastpreprocess PRIVATE ${OpenCV_LIBRARIES} ${pybind11_LIBRARIES})



#############################################
# OMP support
#############################################
find_package(OpenMP REQUIRED)
if(NOT DEFINED OMP_SET AND OPENMP_FOUND)
    message(STATUS "[${PROJECT_NAME}] Using OpenMP")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    set(OMP_SET ON)
endif()

