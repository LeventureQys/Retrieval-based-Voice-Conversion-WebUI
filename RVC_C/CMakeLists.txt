cmake_minimum_required(VERSION 3.15)
project(RVC_ONNX VERSION 1.0.0 LANGUAGES C CXX)

# C/C++ 标准设置
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 输出目录设置
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# 编译选项
if(MSVC)
    add_compile_options(/W4 /utf-8)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
else()
    add_compile_options(-Wall -Wextra -O2)
endif()

# =============================================================================
# 第三方库配置
# =============================================================================

set(THIRD_PARTY_DIR ${CMAKE_SOURCE_DIR}/third_party)

# -----------------------------------------------------------------------------
# ONNX Runtime
# -----------------------------------------------------------------------------
if(WIN32)
    set(ONNXRUNTIME_ROOT ${THIRD_PARTY_DIR}/onnx_lib/onnxruntime-win-x64-1.23.2/onnxruntime-win-x64-1.23.2)
else()
    set(ONNXRUNTIME_ROOT ${THIRD_PARTY_DIR}/onnx_lib/onnxruntime-linux-x64-1.23.2/onnxruntime-linux-x64-1.23.2)
endif()

set(ONNXRUNTIME_INCLUDE_DIRS ${ONNXRUNTIME_ROOT}/include)

if(WIN32)
    set(ONNXRUNTIME_LIBRARIES ${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib)
    set(ONNXRUNTIME_DLL ${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll)
else()
    set(ONNXRUNTIME_LIBRARIES ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so.1.23.2)
endif()

# -----------------------------------------------------------------------------
# KissFFT
# -----------------------------------------------------------------------------
set(KISSFFT_DIR ${THIRD_PARTY_DIR}/kissfft)
set(KISSFFT_INCLUDE_DIRS ${KISSFFT_DIR})

# KissFFT 源文件 (kiss_fftr.c 在根目录)
set(KISSFFT_SOURCES
    ${KISSFFT_DIR}/kiss_fft.c
    ${KISSFFT_DIR}/kiss_fftr.c
)

# -----------------------------------------------------------------------------
# World Vocoder
# -----------------------------------------------------------------------------
set(WORLD_DIR ${THIRD_PARTY_DIR}/World)
set(WORLD_INCLUDE_DIRS ${WORLD_DIR}/src)

# World 源文件
file(GLOB WORLD_SOURCES ${WORLD_DIR}/src/*.cpp)

# -----------------------------------------------------------------------------
# PortAudio (可选，用于实时音频)
# -----------------------------------------------------------------------------
set(PORTAUDIO_DIR ${THIRD_PARTY_DIR}/portaudio)
set(PORTAUDIO_INCLUDE_DIRS ${PORTAUDIO_DIR}/include)

# PortAudio 需要单独编译，这里先设置路径
# 实际使用时需要先编译 PortAudio 库

# -----------------------------------------------------------------------------
# libsndfile (需要编译)
# -----------------------------------------------------------------------------
set(LIBSNDFILE_DIR ${THIRD_PARTY_DIR}/libsndfile)
set(LIBSNDFILE_INCLUDE_DIRS ${LIBSNDFILE_DIR}/include)

# -----------------------------------------------------------------------------
# libresample (高质量重采样库)
# -----------------------------------------------------------------------------
set(LIBRESAMPLE_DIR ${THIRD_PARTY_DIR}/libresample)
set(LIBRESAMPLE_INCLUDE_DIRS ${LIBRESAMPLE_DIR}/include)

# libresample 源文件
set(LIBRESAMPLE_SOURCES
    ${LIBRESAMPLE_DIR}/src/resample.c
    ${LIBRESAMPLE_DIR}/src/resamplesubs.c
    ${LIBRESAMPLE_DIR}/src/filterkit.c
)

# =============================================================================
# 项目源文件
# =============================================================================

# 头文件目录
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${ONNXRUNTIME_INCLUDE_DIRS}
    ${KISSFFT_INCLUDE_DIRS}
    ${WORLD_INCLUDE_DIRS}
    ${WORLD_DIR}/src/world
    ${LIBRESAMPLE_INCLUDE_DIRS}
)

# 核心库源文件
set(RVC_CORE_SOURCES
    src/rvc_onnx.cpp
    src/onnx_inference.cpp
    src/audio_processor.cpp
    src/f0_extractor.cpp
    src/stft.cpp
    src/utils.cpp
    ${KISSFFT_SOURCES}
    ${WORLD_SOURCES}
    ${LIBRESAMPLE_SOURCES}
)

# =============================================================================
# 构建目标
# =============================================================================

# 静态库
add_library(rvc_core STATIC ${RVC_CORE_SOURCES})
target_link_libraries(rvc_core ${ONNXRUNTIME_LIBRARIES})

# 简单测试程序
add_executable(test_onnx_runtime tests/test_onnx_runtime.cpp)
target_link_libraries(test_onnx_runtime rvc_core ${ONNXRUNTIME_LIBRARIES})

# STFT 测试程序
add_executable(test_stft tests/test_stft.cpp)
target_link_libraries(test_stft rvc_core)

# F0 提取测试程序
add_executable(test_f0 tests/test_f0.cpp)
target_link_libraries(test_f0 rvc_core)

# 完整流程测试
add_executable(test_pipeline tests/test_pipeline.cpp)
target_link_libraries(test_pipeline rvc_core ${ONNXRUNTIME_LIBRARIES})

# 合成器测试
add_executable(test_synthesizer tests/test_synthesizer.cpp)
target_link_libraries(test_synthesizer rvc_core ${ONNXRUNTIME_LIBRARIES})

# 完整流程测试 (ContentVec + Synthesizer)
add_executable(test_full_pipeline tests/test_full_pipeline.cpp)
target_link_libraries(test_full_pipeline rvc_core ${ONNXRUNTIME_LIBRARIES})

# 合成器对比测试
add_executable(test_synth_compare tests/test_synth_compare.cpp)
target_link_libraries(test_synth_compare rvc_core ${ONNXRUNTIME_LIBRARIES})

# =============================================================================
# 安装配置
# =============================================================================

# 复制 ONNX Runtime DLL 到输出目录 (Windows)
if(WIN32)
    add_custom_command(TARGET test_onnx_runtime POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${ONNXRUNTIME_DLL}
        $<TARGET_FILE_DIR:test_onnx_runtime>
    )
endif()

# =============================================================================
# 打印配置信息
# =============================================================================

message(STATUS "=== RVC_ONNX Configuration ===")
message(STATUS "ONNX Runtime: ${ONNXRUNTIME_ROOT}")
message(STATUS "KissFFT: ${KISSFFT_DIR}")
message(STATUS "World Vocoder: ${WORLD_DIR}")
message(STATUS "PortAudio: ${PORTAUDIO_DIR}")
message(STATUS "==============================")
