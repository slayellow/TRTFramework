cmake_minimum_required(VERSION 3.5)

project(trtframework LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)
set(TensorRT_DIR "/usr/include/x86_64-linux-gnu")

find_package(CUDA REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS} )
link_directories(${CUDA_LIBRARY_DIRS} )

link_libraries(nvinfer nvonnxparser ${CUDA_LIBRARIES})

add_executable(trtframework main.cpp
    TRT/CTRTEngine.h TRT/CTRTEngine.cpp
    TRT/common.h)

include(GNUInstallDirs)
install(TARGETS trtframework
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
target_link_libraries(trtframework ${CUDA_LIBRARIES})
