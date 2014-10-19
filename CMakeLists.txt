cmake_minimum_required (VERSION 2.6)
PROJECT (KSMs)
# The version number.
SET (Tutorial_VERSION_MAJOR 1)
SET (Tutorial_VERSION_MINOR 0)

# set source list
#SET (SRC main.c)
 
# set linker
# set the below Linker, MacOSx shows gcc-mp-4.5: no input files
#SET (CMAKE_C_LINK_EXECUTABLE /opt/local/bin/mpicc)
# add message
MESSAGE(STATUS "binary dir is" ${PROJECT_BINARY_DIR} )
MESSAGE(STATUS "binary dir is" ${KSMs_BINARY_DIR} )
MESSAGE(STATUS "source dir is" ${PROJECT_SOURCE_DIR} )
MESSAGE(STATUS "output dir is" ${EXECUTABLE_OUTPUT_PATH} )
 
# add the binary tree to the search path for include files
# so that we will find TutorialConfig.h
#include_directories("${PROJECT_BINARY_DIR}")
 
# add the executable
ADD_SUBDIRECTORY(src bin)

# add executable
#ADD_EXECUTABLE(KSMs ${SRC})
