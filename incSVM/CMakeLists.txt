#cmake_minimum_required(VERSION 2.8.3)
#project(LASVM)
#set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
#set(_DEBUG)
#set(CMAKE_BUILD_TYPE debug)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
SET(CMAKE_C_COMPILER g++ )
include_directories(${PROJECT_SOURCE_DIR})

file(GLOB SVMc ${PROJECT_SOURCE_DIR}/incSVM/*.c)
file(GLOB SVMincs ${PROJECT_SOURCE_DIR}/incSVM/*.h)
 
add_library(LaSVM ${SVMc} ${SVMincs})

add_executable(la_svm la_svm.cpp ${SVMc} ${SVMincs})
add_executable(la_incr la_incr.cpp ${SVMc} ${SVMincs})
add_executable(la_test la_test.cpp ${SVMc} ${SVMincs})
add_executable(libsvm2bin libsvm2bin.cpp ${SVMc} ${SVMincs})
add_executable(bin2libsvm bin2libsvm.cpp ${SVMc} ${SVMincs})

