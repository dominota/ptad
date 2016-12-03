cmake_minimum_required(VERSION 2.8.3)
 if(COMMAND cmake_policy)
     cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)
project(PTAD)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
#set(_DEBUG)
set(CMAKE_BUILD_TYPE release)
if(CMAKE_COMPILER_IS_GNUCC)
    ADD_DEFINITIONS ( -std=c++11 )
endif(CMAKE_COMPILER_IS_GNUCC)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "/opt/OpenCV3.0/share/OpenCV")
include("/opt/OpenCV3.0/share/OpenCV/OpenCVConfig.cmake")

find_package(Boost REQUIRED COMPONENTS thread system chrono)
include_directories(${PROJECT_SOURCE_DIR}/include
                    ${OpenCV_INCLUDE_DIRS}
                    ${Boost_INCLUDE_DIRS})


set(CF_CV_EXT_DIR ${PROJECT_SOURCE_DIR}/kcf/3rdparty/cv_ext )
set(CF_PIOTR_DIR  ${PROJECT_SOURCE_DIR}/kcf/3rdparty/piotr)
include_directories(${CF_CV_EXT_DIR}
                    ${CF_PIOTR_DIR}
                    ${CF_PIOTR_DIR}/src)
include_directories(${PROJECT_SOURCE_DIR}/kcf/cf_libs/kcf
                    ${PROJECT_SOURCE_DIR}/kcf/cf_libs/dsst
                    ${PROJECT_SOURCE_DIR}/kcf/cf_libs/common)
add_subdirectory(kcf)



include_directories(${PROJECT_SOURCE_DIR}/scaleCT)
add_subdirectory(scaleCT)


include_directories( ${PROJECT_SOURCE_DIR}/SVM
                     ${PROJECT_SOURCE_DIR}/SVM/blas)
file(GLOB SVMcpp ${PROJECT_SOURCE_DIR}/SVM/*.cpp)
file(GLOB SVMc ${PROJECT_SOURCE_DIR}/SVM/*.c)
file(GLOB blascpp ${PROJECT_SOURCE_DIR}/SVM/blas/*.c)
add_library(SVM    ${SVMcpp} ${SVMc} ${blascpp})

add_subdirectory(incSVM)
include_directories(incSVM)

add_library(patdmodules   src/onlineBoosting.cpp
                   src/onlineMIL.cpp
                   src/haarfeature.cpp
                   src/compressive.cpp
                   src/utils.cpp
                   src/LKTracker.cpp
                   src/Classifier.cpp
                   src/detection.cpp
                   src/system.cpp
                   src/FileFunctions.cpp
                   src/hogfeature
                   src/FHOG.cpp
                   src/fhogtools.cpp
                   src/feature.cpp
                   src/trackerFeature.cpp
                   src/trackerFeatureSet.cpp)

add_executable(ptad src/ptad.cpp)

target_link_libraries(ptad patdmodules compressiveTracking kcf SVM  LaSVM X11 ${OpenCV_LIBS} ${Boost_LIBRARIES})
