/**
 * @file stft.cpp
 * @brief STFT (短时傅里叶变换) 实现
 */

#include "stft.h"
#include "utils.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// KissFFT
#include "kiss_fft.h"
#include "kiss_fftr.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// =============================================================================
// 内部结构定义
// =============================================================================

struct STFTProcessor {
    size_t fft_size;
    size_t hop_size;
    WindowType window_type;

    float* window;
    kiss_fftr_cfg fft_cfg;
    kiss_fftr_cfg ifft_cfg;

    float* fft_input;
    kiss_fft_cpx* fft_output;
};

// =============================================================================
// 窗函数
// =============================================================================

static void create_window(float* window, size_t size, WindowType type) {
    for (size_t i = 0; i < size; i++) {
        double n = (double)i / (size - 1);
        switch (type) {
            case WINDOW_HANN:
                window[i] = (float)(0.5 * (1.0 - cos(2.0 * M_PI * n)));
                break;
            case WINDOW_HAMMING:
                window[i] = (float)(0.54 - 0.46 * cos(2.0 * M_PI * n));
                break;
            case WINDOW_BLACKMAN:
                window[i] = (float)(0.42 - 0.5 * cos(2.0 * M_PI * n) + 0.08 * cos(4.0 * M_PI * n));
                break;
            case WINDOW_RECTANGULAR:
            default:
                window[i] = 1.0f;
                break;
        }
    }
}

// =============================================================================
// STFT 处理器
// =============================================================================

STFTProcessor* stft_processor_create(size_t fft_size, size_t hop_size, WindowType window_type) {
    if (!is_power_of_2(fft_size)) {
        LOG_ERROR("FFT size must be power of 2, got %zu", fft_size);
        return NULL;
    }

    STFTProcessor* processor = (STFTProcessor*)malloc(sizeof(STFTProcessor));
    if (!processor) {
        return NULL;
    }

    processor->fft_size = fft_size;
    processor->hop_size = hop_size;
    processor->window_type = window_type;

    // 创建窗函数
    processor->window = (float*)malloc(fft_size * sizeof(float));
    if (!processor->window) {
        free(processor);
        return NULL;
    }
    create_window(processor->window, fft_size, window_type);

    // 创建FFT配置
    processor->fft_cfg = kiss_fftr_alloc((int)fft_size, 0, NULL, NULL);
    processor->ifft_cfg = kiss_fftr_alloc((int)fft_size, 1, NULL, NULL);

    if (!processor->fft_cfg || !processor->ifft_cfg) {
        if (processor->fft_cfg) kiss_fftr_free(processor->fft_cfg);
        if (processor->ifft_cfg) kiss_fftr_free(processor->ifft_cfg);
        free(processor->window);
        free(processor);
        return NULL;
    }

    // 分配工作缓冲区
    processor->fft_input = (float*)malloc(fft_size * sizeof(float));
    processor->fft_output = (kiss_fft_cpx*)malloc((fft_size / 2 + 1) * sizeof(kiss_fft_cpx));

    if (!processor->fft_input || !processor->fft_output) {
        kiss_fftr_free(processor->fft_cfg);
        kiss_fftr_free(processor->ifft_cfg);
        free(processor->window);
        if (processor->fft_input) free(processor->fft_input);
        if (processor->fft_output) free(processor->fft_output);
        free(processor);
        return NULL;
    }

    LOG_INFO("STFT processor created: fft_size=%zu, hop_size=%zu", fft_size, hop_size);
    return processor;
}

void stft_processor_destroy(STFTProcessor* processor) {
    if (processor) {
        if (processor->fft_cfg) kiss_fftr_free(processor->fft_cfg);
        if (processor->ifft_cfg) kiss_fftr_free(processor->ifft_cfg);
        if (processor->window) free(processor->window);
        if (processor->fft_input) free(processor->fft_input);
        if (processor->fft_output) free(processor->fft_output);
        free(processor);
    }
}

// =============================================================================
// STFT 变换
// =============================================================================

int stft_forward(
    STFTProcessor* processor,
    const float* audio,
    size_t audio_size,
    STFTResult* result
) {
    if (!processor || !audio || !result) {
        return -1;
    }

    size_t fft_size = processor->fft_size;
    size_t hop_size = processor->hop_size;
    size_t num_bins = fft_size / 2 + 1;

    // 计算帧数
    size_t num_frames = 0;
    if (audio_size >= fft_size) {
        num_frames = (audio_size - fft_size) / hop_size + 1;
    }

    if (num_frames == 0) {
        LOG_WARNING("Audio too short for STFT");
        return -2;
    }

    // 分配结果内存
    result->data = (Complex*)malloc(num_frames * num_bins * sizeof(Complex));
    if (!result->data) {
        return -3;
    }

    result->num_frames = num_frames;
    result->num_bins = num_bins;
    result->fft_size = fft_size;
    result->hop_size = hop_size;

    // 逐帧处理
    for (size_t frame = 0; frame < num_frames; frame++) {
        size_t start = frame * hop_size;

        // 应用窗函数
        for (size_t i = 0; i < fft_size; i++) {
            processor->fft_input[i] = audio[start + i] * processor->window[i];
        }

        // 执行FFT
        kiss_fftr(processor->fft_cfg, processor->fft_input, processor->fft_output);

        // 复制结果
        Complex* frame_data = &result->data[frame * num_bins];
        for (size_t i = 0; i < num_bins; i++) {
            frame_data[i].real = processor->fft_output[i].r;
            frame_data[i].imag = processor->fft_output[i].i;
        }
    }

    return 0;
}

int stft_inverse(
    STFTProcessor* processor,
    const STFTResult* stft_data,
    float** audio,
    size_t* audio_size
) {
    if (!processor || !stft_data || !audio || !audio_size) {
        return -1;
    }

    size_t fft_size = processor->fft_size;
    size_t hop_size = processor->hop_size;
    size_t num_frames = stft_data->num_frames;
    size_t num_bins = stft_data->num_bins;

    // 计算输出大小
    *audio_size = (num_frames - 1) * hop_size + fft_size;

    // 分配输出缓冲区
    *audio = (float*)calloc(*audio_size, sizeof(float));
    float* window_sum = (float*)calloc(*audio_size, sizeof(float));

    if (!*audio || !window_sum) {
        if (*audio) free(*audio);
        if (window_sum) free(window_sum);
        return -2;
    }

    // 逐帧处理
    kiss_fft_cpx* freq_data = (kiss_fft_cpx*)malloc(num_bins * sizeof(kiss_fft_cpx));
    float* time_data = (float*)malloc(fft_size * sizeof(float));

    for (size_t frame = 0; frame < num_frames; frame++) {
        size_t start = frame * hop_size;
        const Complex* frame_data = &stft_data->data[frame * num_bins];

        // 准备频域数据
        for (size_t i = 0; i < num_bins; i++) {
            freq_data[i].r = frame_data[i].real;
            freq_data[i].i = frame_data[i].imag;
        }

        // 执行IFFT
        kiss_fftri(processor->ifft_cfg, freq_data, time_data);

        // 归一化并叠加
        for (size_t i = 0; i < fft_size; i++) {
            (*audio)[start + i] += time_data[i] * processor->window[i] / fft_size;
            window_sum[start + i] += processor->window[i] * processor->window[i];
        }
    }

    // 归一化
    for (size_t i = 0; i < *audio_size; i++) {
        if (window_sum[i] > 1e-8f) {
            (*audio)[i] /= window_sum[i];
        }
    }

    free(freq_data);
    free(time_data);
    free(window_sum);

    return 0;
}

void stft_result_free(STFTResult* result) {
    if (result && result->data) {
        free(result->data);
        result->data = NULL;
        result->num_frames = 0;
        result->num_bins = 0;
    }
}

// =============================================================================
// Mel 频谱
// =============================================================================

float hz_to_mel(float freq) {
    return 2595.0f * log10f(1.0f + freq / 700.0f);
}

float mel_to_hz(float mel) {
    return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
}

int mel_filterbank_create(
    int sample_rate,
    size_t fft_size,
    size_t num_mels,
    float fmin,
    float fmax,
    float** filterbank
) {
    if (!filterbank || num_mels == 0 || fft_size == 0) {
        return -1;
    }

    size_t num_bins = fft_size / 2 + 1;

    // 分配滤波器组
    *filterbank = (float*)calloc(num_mels * num_bins, sizeof(float));
    if (!*filterbank) {
        return -2;
    }

    // 计算Mel刻度的边界点
    float mel_min = hz_to_mel(fmin);
    float mel_max = hz_to_mel(fmax);

    float* mel_points = (float*)malloc((num_mels + 2) * sizeof(float));
    float* hz_points = (float*)malloc((num_mels + 2) * sizeof(float));
    size_t* bin_points = (size_t*)malloc((num_mels + 2) * sizeof(size_t));

    for (size_t i = 0; i < num_mels + 2; i++) {
        mel_points[i] = mel_min + (mel_max - mel_min) * i / (num_mels + 1);
        hz_points[i] = mel_to_hz(mel_points[i]);
        bin_points[i] = (size_t)((fft_size + 1) * hz_points[i] / sample_rate);
    }

    // 创建三角滤波器
    for (size_t m = 0; m < num_mels; m++) {
        size_t start = bin_points[m];
        size_t center = bin_points[m + 1];
        size_t end = bin_points[m + 2];

        // 上升沿
        for (size_t k = start; k < center && k < num_bins; k++) {
            (*filterbank)[m * num_bins + k] = (float)(k - start) / (center - start);
        }

        // 下降沿
        for (size_t k = center; k < end && k < num_bins; k++) {
            (*filterbank)[m * num_bins + k] = (float)(end - k) / (end - center);
        }
    }

    free(mel_points);
    free(hz_points);
    free(bin_points);

    return 0;
}

int mel_spectrum_compute(
    const STFTResult* stft_result,
    const float* filterbank,
    size_t num_mels,
    MelSpectrum* mel_spectrum
) {
    if (!stft_result || !filterbank || !mel_spectrum) {
        return -1;
    }

    size_t num_frames = stft_result->num_frames;
    size_t num_bins = stft_result->num_bins;

    // 分配Mel频谱
    mel_spectrum->data = (float*)malloc(num_frames * num_mels * sizeof(float));
    if (!mel_spectrum->data) {
        return -2;
    }

    mel_spectrum->num_frames = num_frames;
    mel_spectrum->num_mels = num_mels;

    // 计算每帧的Mel频谱
    for (size_t frame = 0; frame < num_frames; frame++) {
        const Complex* frame_data = &stft_result->data[frame * num_bins];

        for (size_t m = 0; m < num_mels; m++) {
            float sum = 0.0f;
            for (size_t k = 0; k < num_bins; k++) {
                // 计算幅度
                float mag = sqrtf(frame_data[k].real * frame_data[k].real +
                                  frame_data[k].imag * frame_data[k].imag);
                sum += mag * filterbank[m * num_bins + k];
            }
            mel_spectrum->data[frame * num_mels + m] = sum;
        }
    }

    return 0;
}

void mel_spectrum_free(MelSpectrum* spectrum) {
    if (spectrum && spectrum->data) {
        free(spectrum->data);
        spectrum->data = NULL;
        spectrum->num_frames = 0;
        spectrum->num_mels = 0;
    }
}

// =============================================================================
// 工具函数
// =============================================================================

int stft_magnitude(const STFTResult* stft_result, float** magnitude) {
    if (!stft_result || !magnitude) {
        return -1;
    }

    size_t total = stft_result->num_frames * stft_result->num_bins;
    *magnitude = (float*)malloc(total * sizeof(float));
    if (!*magnitude) {
        return -2;
    }

    for (size_t i = 0; i < total; i++) {
        *magnitude[i] = sqrtf(stft_result->data[i].real * stft_result->data[i].real +
                              stft_result->data[i].imag * stft_result->data[i].imag);
    }

    return 0;
}

int stft_phase(const STFTResult* stft_result, float** phase) {
    if (!stft_result || !phase) {
        return -1;
    }

    size_t total = stft_result->num_frames * stft_result->num_bins;
    *phase = (float*)malloc(total * sizeof(float));
    if (!*phase) {
        return -2;
    }

    for (size_t i = 0; i < total; i++) {
        (*phase)[i] = atan2f(stft_result->data[i].imag, stft_result->data[i].real);
    }

    return 0;
}
