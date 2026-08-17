cmake_minimum_required(VERSION 3.21)
project(stipple LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_definitions($<$<CONFIG:Release>:NDEBUG>)

# Flat bin/, including for multi-config generators that would otherwise nest
# the output under bin/Release/.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
foreach(config ${CMAKE_CONFIGURATION_TYPES})
  string(TOUPPER ${config} config_upper)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${config_upper} ${CMAKE_SOURCE_DIR}/bin)
endforeach()

option(STIPPLE_ENABLE_GPU "Build GPU support when a CUDA toolchain is available" ON)
set(STIPPLE_GPU_ARCH "86" CACHE STRING "CUDA compute capability (75=Turing, 86=Ampere, 89=Ada)")

find_package(OpenMP REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc)

# CUDA is detected, never required: a machine without a usable nvcc still
# configures and builds the CPU-only binary instead of failing outright.
set(STIPPLE_CUDA_AVAILABLE FALSE)
if(STIPPLE_ENABLE_GPU)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(STIPPLE_CUDA_AVAILABLE TRUE)
  else()
    message(WARNING "No usable CUDA toolchain found; building without GPU support.")
  endif()
endif()

set(STIPPLE_CORE_SOURCES
    src/image/image.cpp
    src/algorithm/common.cpp
    src/algorithm/serial.cpp
    src/algorithm/cpu_parallel.cpp
    src/algorithm/cpu_simd.cpp
    src/benchmark/benchmark.cpp
    src/utils/cli.cpp
    src/utils/runner.cpp
    src/utils/interactive.cpp)

if(STIPPLE_CUDA_AVAILABLE)
  list(APPEND STIPPLE_CORE_SOURCES src/algorithm/gpu.cu)
endif()

# AVX2 codegen scoped to the one file that uses AVX2 intrinsics -- everything
# else in the binary must stay runnable on a pre-AVX2 CPU, since
# simdAvx2Available() is a *runtime* CPUID check, not a build-time guarantee.
set_source_files_properties(src/algorithm/cpu_simd.cpp PROPERTIES COMPILE_OPTIONS
    "$<$<CXX_COMPILER_ID:MSVC>:/arch:AVX2>;$<$<CXX_COMPILER_ID:GNU,Clang>:-mavx2>")

# Applies the include/link/CUDA/DLL-copy setup shared by every front end
# (CLI+TUI binary, GUI binary) to one target, so adding a new front end never
# means re-deriving this by hand.
function(stipple_configure_target target_name)
  target_include_directories(${target_name} PRIVATE ${CMAKE_SOURCE_DIR}/src ${OpenCV_INCLUDE_DIRS})
  target_link_libraries(${target_name} PRIVATE OpenMP::OpenMP_CXX ${OpenCV_LIBS})
  target_compile_options(${target_name} PRIVATE $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra>)

  if(STIPPLE_CUDA_AVAILABLE)
    target_compile_definitions(${target_name} PRIVATE STIPPLE_CUDA_ENABLED)
    # A real cubin for the installed GPU, so the driver never has to JIT PTX
    # from a toolkit newer than itself.
    set_target_properties(${target_name} PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_ARCHITECTURES "${STIPPLE_GPU_ARCH}-real;${STIPPLE_GPU_ARCH}-virtual")
  endif()

  # Without this the program cannot find OpenCV's DLL and dies on launch with
  # a bare 0xC0000135 and no message.
  if(WIN32)
    add_custom_command(TARGET ${target_name} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_RUNTIME_DLLS:${target_name}> $<TARGET_FILE_DIR:${target_name}>
        COMMAND_EXPAND_LISTS)
  endif()
endfunction()

add_executable(stipple src/main.cpp ${STIPPLE_CORE_SOURCES})
stipple_configure_target(stipple)

add_custom_target(run
    COMMAND $<TARGET_FILE:stipple>
    DEPENDS stipple
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    USES_TERMINAL)

# GUI front end: Win32 only (no cross-platform toolkit dependency), reusing
# the same STIPPLE_CORE_SOURCES backend as the CLI/TUI binary.
if(WIN32)
  add_executable(stipple_gui WIN32 src/gui/win32_gui.cpp ${STIPPLE_CORE_SOURCES})
  target_link_libraries(stipple_gui PRIVATE comctl32 comdlg32 user32 gdi32)
  target_compile_definitions(stipple_gui PRIVATE UNICODE _UNICODE)
  stipple_configure_target(stipple_gui)

  add_custom_target(run-gui
      COMMAND $<TARGET_FILE:stipple_gui>
      DEPENDS stipple_gui
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
      USES_TERMINAL)
endif()
