/**
 * @file audio_processor.h
 * @brief 音频处理模块
 */

#ifndef AUDIO_PROCESSOR_H
#define AUDIO_PROCESSOR_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// 类型定义
// =============================================================================

/** 音频处理器句柄 */
typedef struct AudioProcessor AudioProcessor;

/** 音频格式 */
typedef struct {
    int sample_rate;
    int channels;
    int bits_per_sample;
} AudioFormat;

/** 音频缓冲区 */
typedef struct {
    float* data;
    size_t size;
    size_t capacity;
    AudioFormat format;
} AudioBuffer;

// =============================================================================
// 音频处理器
// =============================================================================

/**
 * @brief 创建音频处理器
 * @param sample_rate 采样率
 * @return 处理器句柄
 */
AudioProcessor* audio_processor_create(int sample_rate);

/**
 * @brief 销毁音频处理器
 * @param processor 处理器句柄
 */
void audio_processor_destroy(AudioProcessor* processor);

// =============================================================================
// 音频I/O
// =============================================================================

/**
 * @brief 从文件加载音频
 * @param filepath 文件路径
 * @param buffer 输出缓冲区
 * @param format 输出格式信息
 * @return 0成功，非0失败
 */
int audio_load_file(const char* filepath, AudioBuffer* buffer, AudioFormat* format);

/**
 * @brief 保存音频到文件
 * @param filepath 文件路径
 * @param buffer 音频数据
 * @param format 格式信息
 * @return 0成功，非0失败
 */
int audio_save_file(const char* filepath, const AudioBuffer* buffer, const AudioFormat* format);

/**
 * @brief 释放音频缓冲区
 * @param buffer 缓冲区
 */
void audio_buffer_free(AudioBuffer* buffer);

/**
 * @brief 创建音频缓冲区
 * @param capacity 容量
 * @return 缓冲区
 */
AudioBuffer audio_buffer_create(size_t capacity);

// =============================================================================
// 音频处理
// =============================================================================

/**
 * @brief 重采样
 * @param input 输入数据
 * @param input_size 输入大小
 * @param src_rate 源采样率
 * @param dst_rate 目标采样率
 * @param output 输出数据 (由函数分配)
 * @param output_size 输出大小
 * @return 0成功，非0失败
 */
int audio_resample(
    const float* input,
    size_t input_size,
    int src_rate,
    int dst_rate,
    float** output,
    size_t* output_size
);

/**
 * @brief 音频归一化
 * @param audio 音频数据 (原地修改)
 * @param size 数据大小
 * @param target_db 目标分贝 (默认 -6.0)
 */
void audio_normalize(float* audio, size_t size, float target_db);

/**
 * @brief 音频预处理 (归一化 + 去直流)
 * @param processor 处理器句柄
 * @param input 输入数据
 * @param input_size 输入大小
 * @param output 输出数据
 * @param output_size 输出大小
 * @return 0成功，非0失败
 */
int audio_preprocess(
    AudioProcessor* processor,
    const float* input,
    size_t input_size,
    float* output,
    size_t* output_size
);

/**
 * @brief 静音检测
 * @param audio 音频数据
 * @param size 数据大小
 * @param threshold_db 阈值分贝
 * @return 1为静音，0为非静音
 */
int audio_is_silent(const float* audio, size_t size, float threshold_db);

/**
 * @brief 计算RMS能量
 * @param audio 音频数据
 * @param size 数据大小
 * @return RMS值
 */
float audio_rms(const float* audio, size_t size);

/**
 * @brief 计算峰值
 * @param audio 音频数据
 * @param size 数据大小
 * @return 峰值
 */
float audio_peak(const float* audio, size_t size);

#ifdef __cplusplus
}
#endif

#endif // AUDIO_PROCESSOR_H
