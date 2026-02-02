/**
 * @file stft.h
 * @brief STFT (短时傅里叶变换) 模块
 */

#ifndef STFT_H
#define STFT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// 类型定义
// =============================================================================

/** STFT 处理器句柄 */
typedef struct STFTProcessor STFTProcessor;

/** 复数结构 */
typedef struct {
    float real;
    float imag;
} Complex;

/** STFT 结果 */
typedef struct {
    Complex* data;          /**< 复数频谱数据 [num_frames x num_bins] */
    size_t num_frames;      /**< 帧数 */
    size_t num_bins;        /**< 频率bin数 (fft_size/2 + 1) */
    size_t fft_size;        /**< FFT大小 */
    size_t hop_size;        /**< 跳跃大小 */
} STFTResult;

/** Mel 频谱结果 */
typedef struct {
    float* data;            /**< Mel频谱数据 [num_frames x num_mels] */
    size_t num_frames;      /**< 帧数 */
    size_t num_mels;        /**< Mel频带数 */
} MelSpectrum;

/** 窗函数类型 */
typedef enum {
    WINDOW_HANN = 0,
    WINDOW_HAMMING = 1,
    WINDOW_BLACKMAN = 2,
    WINDOW_RECTANGULAR = 3,
} WindowType;

// =============================================================================
// STFT 处理器
// =============================================================================

/**
 * @brief 创建STFT处理器
 * @param fft_size FFT大小 (通常为2048)
 * @param hop_size 跳跃大小 (通常为512)
 * @param window_type 窗函数类型
 * @return 处理器句柄
 */
STFTProcessor* stft_processor_create(size_t fft_size, size_t hop_size, WindowType window_type);

/**
 * @brief 销毁STFT处理器
 * @param processor 处理器句柄
 */
void stft_processor_destroy(STFTProcessor* processor);

// =============================================================================
// STFT 变换
// =============================================================================

/**
 * @brief 执行STFT变换
 * @param processor 处理器句柄
 * @param audio 输入音频
 * @param audio_size 音频大小
 * @param result 输出结果 (由函数分配)
 * @return 0成功，非0失败
 */
int stft_forward(
    STFTProcessor* processor,
    const float* audio,
    size_t audio_size,
    STFTResult* result
);

/**
 * @brief 执行ISTFT逆变换
 * @param processor 处理器句柄
 * @param stft_data STFT数据
 * @param audio 输出音频 (由函数分配)
 * @param audio_size 输出大小
 * @return 0成功，非0失败
 */
int stft_inverse(
    STFTProcessor* processor,
    const STFTResult* stft_data,
    float** audio,
    size_t* audio_size
);

/**
 * @brief 释放STFT结果
 * @param result STFT结果
 */
void stft_result_free(STFTResult* result);

// =============================================================================
// Mel 频谱
// =============================================================================

/**
 * @brief 创建Mel滤波器组
 * @param sample_rate 采样率
 * @param fft_size FFT大小
 * @param num_mels Mel频带数
 * @param fmin 最低频率 (Hz)
 * @param fmax 最高频率 (Hz)
 * @param filterbank 输出滤波器组 [num_mels x (fft_size/2+1)]
 * @return 0成功，非0失败
 */
int mel_filterbank_create(
    int sample_rate,
    size_t fft_size,
    size_t num_mels,
    float fmin,
    float fmax,
    float** filterbank
);

/**
 * @brief 计算Mel频谱
 * @param stft_result STFT结果
 * @param filterbank Mel滤波器组
 * @param num_mels Mel频带数
 * @param mel_spectrum 输出Mel频谱
 * @return 0成功，非0失败
 */
int mel_spectrum_compute(
    const STFTResult* stft_result,
    const float* filterbank,
    size_t num_mels,
    MelSpectrum* mel_spectrum
);

/**
 * @brief 释放Mel频谱
 * @param spectrum Mel频谱
 */
void mel_spectrum_free(MelSpectrum* spectrum);

// =============================================================================
// 工具函数
// =============================================================================

/**
 * @brief 频率转Mel
 * @param freq 频率 (Hz)
 * @return Mel值
 */
float hz_to_mel(float freq);

/**
 * @brief Mel转频率
 * @param mel Mel值
 * @return 频率 (Hz)
 */
float mel_to_hz(float mel);

/**
 * @brief 计算幅度谱
 * @param stft_result STFT结果
 * @param magnitude 输出幅度谱 [num_frames x num_bins]
 * @return 0成功，非0失败
 */
int stft_magnitude(const STFTResult* stft_result, float** magnitude);

/**
 * @brief 计算相位谱
 * @param stft_result STFT结果
 * @param phase 输出相位谱 [num_frames x num_bins]
 * @return 0成功，非0失败
 */
int stft_phase(const STFTResult* stft_result, float** phase);

#ifdef __cplusplus
}
#endif

#endif // STFT_H
