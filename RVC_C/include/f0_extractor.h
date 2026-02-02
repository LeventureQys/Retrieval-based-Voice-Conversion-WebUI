/**
 * @file f0_extractor.h
 * @brief F0 (基频) 提取模块
 */

#ifndef F0_EXTRACTOR_H
#define F0_EXTRACTOR_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// 类型定义
// =============================================================================

/** F0 提取器句柄 */
typedef struct F0Extractor F0Extractor;

/** F0 提取方法 */
typedef enum {
    F0_METHOD_HARVEST = 0,  /**< Harvest 算法 (World) */
    F0_METHOD_DIO = 1,      /**< DIO 算法 (World) */
} F0Method;

/** F0 提取结果 */
typedef struct {
    double* f0;             /**< F0值数组 (Hz) */
    double* time_axis;      /**< 时间轴数组 (秒) */
    size_t length;          /**< 数组长度 */
    int sample_rate;        /**< 采样率 */
    double frame_period;    /**< 帧周期 (毫秒) */
} F0Result;

/** F0 提取参数 */
typedef struct {
    double f0_floor;        /**< 最低F0 (Hz, 默认: 71.0) */
    double f0_ceil;         /**< 最高F0 (Hz, 默认: 800.0) */
    double frame_period;    /**< 帧周期 (毫秒, 默认: 5.0) */
    int speed;              /**< 速度参数 (1-12, 仅Harvest, 默认: 1) */
} F0Params;

// =============================================================================
// F0 提取器
// =============================================================================

/**
 * @brief 创建F0提取器
 * @param method 提取方法
 * @param sample_rate 采样率
 * @return 提取器句柄
 */
F0Extractor* f0_extractor_create(F0Method method, int sample_rate);

/**
 * @brief 销毁F0提取器
 * @param extractor 提取器句柄
 */
void f0_extractor_destroy(F0Extractor* extractor);

/**
 * @brief 设置F0提取参数
 * @param extractor 提取器句柄
 * @param params 参数
 */
void f0_extractor_set_params(F0Extractor* extractor, const F0Params* params);

/**
 * @brief 获取默认参数
 * @return 默认参数
 */
F0Params f0_default_params(void);

// =============================================================================
// F0 提取
// =============================================================================

/**
 * @brief 提取F0
 * @param extractor 提取器句柄
 * @param audio 输入音频
 * @param audio_size 音频大小
 * @param result 输出结果
 * @return 0成功，非0失败
 */
int f0_extract(
    F0Extractor* extractor,
    const double* audio,
    size_t audio_size,
    F0Result* result
);

/**
 * @brief 提取F0 (float版本)
 * @param extractor 提取器句柄
 * @param audio 输入音频 (float)
 * @param audio_size 音频大小
 * @param result 输出结果
 * @return 0成功，非0失败
 */
int f0_extract_float(
    F0Extractor* extractor,
    const float* audio,
    size_t audio_size,
    F0Result* result
);

/**
 * @brief 释放F0结果
 * @param result F0结果
 */
void f0_result_free(F0Result* result);

// =============================================================================
// F0 后处理
// =============================================================================

/**
 * @brief F0平滑处理
 * @param f0 F0数组 (原地修改)
 * @param length 数组长度
 * @param window_size 平滑窗口大小
 */
void f0_smooth(double* f0, size_t length, int window_size);

/**
 * @brief F0中值滤波
 * @param f0 F0数组 (原地修改)
 * @param length 数组长度
 * @param window_size 窗口大小
 */
void f0_median_filter(double* f0, size_t length, int window_size);

/**
 * @brief F0插值 (填补无声段)
 * @param f0 F0数组 (原地修改)
 * @param length 数组长度
 */
void f0_interpolate(double* f0, size_t length);

/**
 * @brief F0 resize (调整到目标长度)
 * @param f0 输入F0数组
 * @param src_length 源长度
 * @param dst_length 目标长度
 * @param output 输出数组 (需要预分配)
 */
void f0_resize(const double* f0, size_t src_length, size_t dst_length, double* output);

/**
 * @brief F0音高偏移
 * @param f0 F0数组 (原地修改)
 * @param length 数组长度
 * @param semitones 半音数
 */
void f0_shift_pitch(double* f0, size_t length, double semitones);

// =============================================================================
// 工具函数
// =============================================================================

/**
 * @brief 频率转MIDI音符
 * @param freq 频率 (Hz)
 * @return MIDI音符号
 */
double freq_to_midi(double freq);

/**
 * @brief MIDI音符转频率
 * @param midi MIDI音符号
 * @return 频率 (Hz)
 */
double midi_to_freq(double midi);

/**
 * @brief F0转Mel刻度
 * @param f0 F0值 (Hz)
 * @return Mel值
 */
double f0_to_mel(double f0);

/**
 * @brief Mel刻度转F0
 * @param mel Mel值
 * @return F0值 (Hz)
 */
double mel_to_f0(double mel);

#ifdef __cplusplus
}
#endif

#endif // F0_EXTRACTOR_H
