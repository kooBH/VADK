# CMAKE version of Ubuntu 16.04 is 3.5.
cmake_minimum_required(VERSION 3.5)
project(GPV)

set(SRC)
set(INCL)
set(LINKLIBS)
set(COMPILE_OPTION)

set(PATH_TRT  D:/TensorRT/TensorRT-8.2.2.1/)

set(PATH_CUDA C:/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v11.5/)

find_package(CUDA REQUIRED)
#find_package(TensorRT REQUIRED)

include(lib/WAV/WAV.cmake)
include(lib/STFT/cpp/STFT_cpp.cmake)
include(lib/STFT/mel/mel.cmake)

list(APPEND SRC
	main.cpp
  )

list(APPEND INCL
  ${CMAKE_CURRENT_LIST_DIR}/.
  ${PATH_TRT}/include
  )
  
list(APPEND LINKLIBS
	${PATH_TRT}/lib/nvinfer.lib
	${PATH_TRT}/lib/nvonnxparser.lib
  )
  
if(MSVC)
  list(APPEND COMPILE_OPTION
  
    )
else()
  list(APPEND COMPILE_OPTION
    )
endif()

message(${CUDA_LIBRARIES})

add_executable(${PROJECT_NAME} ${SRC})
target_include_directories(${PROJECT_NAME}    PRIVATE    ${INCL}  ${CUDA_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${LINKLIBS} ${CUDA_LIBRARIES}  ${TensorRT_LIBRARIES})
target_compile_options(${PROJECT_NAME} PRIVATE ${COMPILE_OPTION} )
set_target_properties(${PROJECT_NAME} PROPERTIES  CXX_STANDARD 17 )