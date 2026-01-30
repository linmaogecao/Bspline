cmake_minimum_required(VERSION 3.16)
project(spline_fitting)

# 1. Set C++ Standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(PCL_DIR "/opt/pcl-1.13/share/pcl-1.13")
find_package(PCL REQUIRED COMPONENTS common io)
add_definitions(${PCL_DEFINITIONS})
link_directories(${PCL_LIBRARY_DIRS})

# 4. Include Directories
include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/core
        ${EIGEN3_INCLUDE_DIRS}
        ${ANN_INCLUDE_DIR}
        ${PCL_INCLUDE_DIRS}  # <--- 新增: PCL 头文件路径
        ${CERES_INCLUDE_DIRS}
)

# 2. Find Eigen3
# On Ubuntu, this is usually installed via: sudo apt install libeigen3-dev
find_package(Eigen3 REQUIRED)
if(EIGEN3_FOUND)
    message(STATUS "Found Eigen3: ${EIGEN3_INCLUDE_DIRS}")
else()
    # Fallback to standard Ubuntu path if find_package fails
    set(EIGEN3_INCLUDE_DIRS "/usr/include/eigen3")
endif()

find_package(Ceres REQUIRED)

find_path(ANN_INCLUDE_DIR NAMES ANN/ANN.h PATHS /usr/include /usr/local/include)

# 尝试寻找库文件，增加小写 "ann" 的支持，并显式搜索 Ubuntu 的库路径
find_library(ANN_LIBRARY 
    NAMES ANN ann  # 同时搜索大写和小写
    PATHS 
        /usr/lib 
        /usr/local/lib 
        /usr/lib/x86_64-linux-gnu  # Ubuntu 64位库的标准存放地
)

if(ANN_INCLUDE_DIR AND ANN_LIBRARY)
    message(STATUS "Found ANN include: ${ANN_INCLUDE_DIR}")
    message(STATUS "Found ANN library: ${ANN_LIBRARY}")
else()
    message(FATAL_ERROR "ANN library not found even though apt says it is installed!")
endif()

# 4. Include Directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/core
    ${EIGEN3_INCLUDE_DIRS}
    ${ANN_INCLUDE_DIR}
    ${PCL_INCLUDE_DIRS}
    ${CERES_INCLUDE_DIRS}
)

# 5. Source Files
set(SOURCE_FILES
    core/BSpline.cpp
    core/BSplineSurface.cpp
    core/Spline_curve_fitting.cpp
    readWrite.cpp
    main.cpp
)

# 6. Headers (Optional, for IDE visibility)
set(HEADER_FILES
    core/BSpline.h
    core/Spline_curve_fitting.h
    readWrite.h
        core/BSplineSDMErr.h
)

# 7. Create Executable
add_executable(spline_fitting_test ${SOURCE_FILES} ${HEADER_FILES})

# 8. Link Libraries
target_link_libraries(spline_fitting_test
    ${ANN_LIBRARY}
    ${PCL_LIBRARIES}
    Ceres::ceres
)
