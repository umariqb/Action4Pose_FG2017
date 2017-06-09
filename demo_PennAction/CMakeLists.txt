# see http://www.cmake.org/ for a documentation about CMake

cmake_minimum_required(VERSION 2.6)
project (demo_PennAction)



######################### Set Module Path ######################### 
############ Path where FindLib.cmake files are located ###########
###################################################################
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

####################### CUSTOM_LIBRARY_PATH ####################### 
#########  Adjust this path if local libraries are used  ##########
###################################################################
set(CUSTOM_LIBRARY_PATH "")

################# Caffe Path ######################################
####### Adjust this path according to your caffe location #########
###################################################################
set(CAFFE_PATH "${CMAKE_SOURCE_DIR}/../caffe")

set(CMAKE_CXX_FLAGS "-std=c++0x")				# New C11


#set(CMAKE_BUILD_TYPE debug)					# choose if you want to compile with or without debugging flags,
set(CMAKE_BUILD_TYPE release)					# uncomment only one of the following two commands at a time

################################## Find required packages ############################
########################### Loading the required packages ############################ 
######################################################################################

##### OpenCV
find_package( OpenCV REQUIRED )
##### Boost
SET(Boost_USE_STATIC_LIBS OFF) 
SET(Boost_USE_MULTITHREADED ON)  
set(Boost_USE_STATIC_RUNTIME OFF) 
find_package(Boost 1.55.0 COMPONENTS date_time iostreams program_options regex serialization system thread filesystem REQUIRED)

##### Cuda
find_package(CUDA REQUIRED)


##### Caffe
find_package(Caffe)
if (Caffe_FOUND)
     get_property(Caffe_INCLUDE_DIRS TARGET caffe PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
     get_property(Caffe_DEFINITIONS TARGET caffe PROPERTY INTERFACE_DEFINITIONS)
    message(STATUS "Found Caffe. (${Caffe_INCLUDE_DIRS})")
else()
    message(STATUS "Caffe not found.")
endif()



####################################### GFlags #######################################  
find_package(GFlags REQUIRED)

######################################## GLOG ######################################## 
find_package(GLOG REQUIRED)

######################################## GLOG ######################################## 
find_package(Config++ REQUIRED)

############################ Include Required Directories ############################ 			
include_directories(${PROJECT_SOURCE_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIR})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${GFLAGS_INCLUDE_DIRS})
include_directories(${GLOG_INCLUDE_DIRS})
include_directories(${Caffe_INCLUDE_DIRS})
include_directories(${CONFIG++_INCLUDE_DIR})
add_definitions(${Caffe_DEFINITIONS})


set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin/)			# Binary file location
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})			# Exectuable file location for main and subprojects 
set(LIBRARY_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/lib)		# Library file location for main and subprojects

include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/../)


############################## Adding Source Files ###################################
######################################################################################
add_executable(demo_PennAction 
${PROJECT_SOURCE_DIR}/demo_PennAction.cpp  
${PROJECT_SOURCE_DIR}/../cpp/body_pose/clustering/body_clustering.cpp   
${PROJECT_SOURCE_DIR}/../cpp/body_pose/body_part_sample.cpp 
${PROJECT_SOURCE_DIR}/../cpp/body_pose/clustering/clustering_features.cpp 
${PROJECT_SOURCE_DIR}/../cpp/body_pose/utils.cpp
${PROJECT_SOURCE_DIR}/../cpp/learning/forest/forest.cpp
${PROJECT_SOURCE_DIR}/../cpp/learning/pictorial_structure/inferenz.cpp
${PROJECT_SOURCE_DIR}/../cpp/learning/pictorial_structure/learn_model_parameter.cpp
${PROJECT_SOURCE_DIR}/../cpp/learning/pictorial_structure/part.cpp
${PROJECT_SOURCE_DIR}/../cpp/learning/pictorial_structure/utils.cpp
${PROJECT_SOURCE_DIR}/../cpp/third_party/hog/hog.cpp
${PROJECT_SOURCE_DIR}/../cpp/utils/file_utils.cpp
${PROJECT_SOURCE_DIR}/../cpp/utils/image_file_utils.cpp
${PROJECT_SOURCE_DIR}/../cpp/utils/serialization/serialization.cpp
${PROJECT_SOURCE_DIR}/../cpp/utils/string_utils.cpp
${PROJECT_SOURCE_DIR}/../cpp/utils/system_utils.cpp
${PROJECT_SOURCE_DIR}/../cpp/utils/threading.cpp
${PROJECT_SOURCE_DIR}/../cpp/fashion/skin/SkinColorModelGaussian.cc
${PROJECT_SOURCE_DIR}/../cpp/fashion/skin/SkinSegmentation.cc
${PROJECT_SOURCE_DIR}/../cpp/vision/features/bow_extractor.cpp
${PROJECT_SOURCE_DIR}/../cpp/vision/features/feature_channels/feature_channel_factory.cpp
${PROJECT_SOURCE_DIR}/../cpp/vision/features/feature_channels/hog_extractor.cpp
${PROJECT_SOURCE_DIR}/../cpp/vision/features/simple_feature.cpp
${PROJECT_SOURCE_DIR}/../cpp/vision/features/global/gist.cpp
${PROJECT_SOURCE_DIR}/../cpp/vision/features/local_word_container.cpp
${PROJECT_SOURCE_DIR}/../cpp/vision/features/low_level/color.cpp
${PROJECT_SOURCE_DIR}/../cpp/vision/features/low_level/dense_surf.cpp
${PROJECT_SOURCE_DIR}/../cpp/vision/features/low_level/hog.cpp
${PROJECT_SOURCE_DIR}/../cpp/vision/features/low_level/holbp.cpp
${PROJECT_SOURCE_DIR}/../cpp/vision/features/low_level/lbp.cpp
${PROJECT_SOURCE_DIR}/../cpp/vision/features/low_level/low_level_feature_extractor.cpp
${PROJECT_SOURCE_DIR}/../cpp/vision/features/low_level/self_similarity.cpp
${PROJECT_SOURCE_DIR}/../cpp/vision/features/low_level/surf.cpp
${PROJECT_SOURCE_DIR}/../cpp/vision/features/low_level_features.cpp
${PROJECT_SOURCE_DIR}/../cpp/vision/features/spmbow_extractor.cpp
${PROJECT_SOURCE_DIR}/../cpp/vision/geometry_utils.cpp
${PROJECT_SOURCE_DIR}/../cpp/vision/image_utils.cpp
${PROJECT_SOURCE_DIR}/../cpp/vision/mean_shift.cpp
${PROJECT_SOURCE_DIR}/../cpp/vision/min_max_filter.cpp
${PROJECT_SOURCE_DIR}/../cpp/third_party/kmeans/kmeans.cpp
${PROJECT_SOURCE_DIR}/../cpp/vision/features/cnn/cnn_features.cpp 
${PROJECT_SOURCE_DIR}/../cpp/vision/features/cnn/caffe_utils.cpp 
${PROJECT_SOURCE_DIR}/../cpp/utils/pyramid_stitcher/pyramid_stitcher.cpp 
${PROJECT_SOURCE_DIR}/../cpp/utils/pyramid_stitcher/image_pyramid.cpp 
${PROJECT_SOURCE_DIR}/../cpp/utils/pyramid_stitcher/patchwork.cpp 
${PROJECT_SOURCE_DIR}/../cpp/utils/pyramid_stitcher/Rectangle.cpp
)	


target_link_libraries( demo_PennAction ${PROJECT_INCLUDE_DIR}  ${OpenCV_LIBS}  ${Boost_LIBRARIES} ${GFLAGS_LIBRARIES} ${GLOG_LIBRARIES} ${CUDA_LIBRARIES} ${Caffe_LIBRARIES} ${CONFIG++_LIBRARY})		

