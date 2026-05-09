cmake_minimum_required (VERSION 3.18)

if (BUILDING_RUNTIME)
    message(FATAL_ERROR "ENABLE_CUDA_SIMULATOR_RUNTIME is only supported for simulator builds. Configure with -DBUILDING_RUNTIME=OFF or disable -DENABLE_CUDA_SIMULATOR_RUNTIME=OFF.")
endif()

set(NNCASE_TRITON_PYTHON_DEFAULT "")
if (DEFINED ENV{NNCASE_TRITON_PYTHON} AND NOT "$ENV{NNCASE_TRITON_PYTHON}" STREQUAL "")
    set(NNCASE_TRITON_PYTHON_DEFAULT "$ENV{NNCASE_TRITON_PYTHON}")
elseif (EXISTS "${CMAKE_SOURCE_DIR}/zsy-nncase/bin/python")
    set(NNCASE_TRITON_PYTHON_DEFAULT "${CMAKE_SOURCE_DIR}/zsy-nncase/bin/python")
elseif (Python3_EXECUTABLE)
    set(NNCASE_TRITON_PYTHON_DEFAULT "${Python3_EXECUTABLE}")
endif()

if (NNCASE_TRITON_PYTHON_DEFAULT AND NOT Python3_EXECUTABLE)
    set(Python3_EXECUTABLE "${NNCASE_TRITON_PYTHON_DEFAULT}" CACHE FILEPATH "Python executable used by the CUDA simulator runtime")
endif()

find_package(CUDAToolkit QUIET)
if (NOT CUDAToolkit_FOUND OR NOT TARGET CUDA::cuda_driver)
    message(FATAL_ERROR "ENABLE_CUDA_SIMULATOR_RUNTIME=ON requires the CUDA toolkit and CUDA driver import library. Install CUDA toolkit headers/libraries or configure with -DENABLE_CUDA_SIMULATOR_RUNTIME=OFF.")
endif()

find_package(Python3 COMPONENTS Interpreter Development.Embed REQUIRED)
if (NOT NNCASE_TRITON_PYTHON_DEFAULT)
    set(NNCASE_TRITON_PYTHON_DEFAULT "${Python3_EXECUTABLE}")
endif()

set(SRCS
    runtime_rdata_layout.cpp
    runtime_function.cpp
    runtime_module.cpp)

add_library(simulator_cuda SHARED ${SRCS})
target_link_libraries(simulator_cuda
    PRIVATE
        nncaseruntime
        CUDA::cuda_driver
        Python3::Python
        fmt::fmt
        nlohmann_json::nlohmann_json)
target_compile_definitions(simulator_cuda
    PRIVATE
        -DNNCASE_DLL
        -DNNCASE_SIMULATOR
        NNCASE_TRITON_PYTHON_DEFAULT="${NNCASE_TRITON_PYTHON_DEFAULT}")
set_target_properties(simulator_cuda PROPERTIES
    OUTPUT_NAME "nncase.simulator.cuda"
    POSITION_INDEPENDENT_CODE ON)

add_executable(simulator_cuda_rdata_layout_test EXCLUDE_FROM_ALL
    runtime_rdata_layout_test.cpp
    runtime_rdata_layout.cpp)
target_compile_definitions(simulator_cuda_rdata_layout_test
    PRIVATE
        -DNNCASE_DLL
        -DNNCASE_SIMULATOR)

install(TARGETS simulator_cuda
    COMPONENT nncase-runtime
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin)
