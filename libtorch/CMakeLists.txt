cmake_minimum_required(VERSION 3.5 )
project(GPV)

set(LINKLIBS)
set(COMPILE_OPTION)
set(SRC)
set(INCL)

#set(CMAKE_PREFIX_PATH lib/libtorch/share/cmake/Torch/)
set(CMAKE_PREFIX_PATH lib/libtorch)
message(STATUS ${CMAKE_PREFIX_PATH})


### LibTorchPath ###
#include(lib/libtorch/share/cmake/Torch/TorchConfig.cmake)
find_package(Torch REQUIRED)

include(lib/WAV/WAV.cmake)
include(lib/STFT/cpp/STFT_cpp.cmake)
include(lib/STFT/mel/mel.cmake)

list(APPEND SRC
		GPV.h
)

list(APPEND INCL

)

list(APPEND LINKLIBS 
  )

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(${PROJECT_NAME} test.cpp ${SRC})
target_include_directories(${PROJECT_NAME}	PRIVATE	${INCL}  )
target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES} ${LINKLIBS} )
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)