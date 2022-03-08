cmake_minimum_required(VERSION 3.4 )
project(cifar)

### LibTorchPath ###
include(lib/libtorch/share/cmake/Torch/TorchConfig.cmake)

list(APPEND INCL
)

list(APPEND LINKLIBS 
  )

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(${PROJECT_NAME} main.cpp)
target_include_directories(${PROJECT_NAME}	PRIVATE	${INCL}  )
target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES} ${LINKLIBS} )
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 14)