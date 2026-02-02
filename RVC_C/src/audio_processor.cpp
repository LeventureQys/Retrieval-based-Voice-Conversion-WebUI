/**
 * @file audio_processor.cpp
 * @brief 音频处理模块实现
 */

#include "audio_processor.h"
#include "utils.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cstdio>

// 高质量重采样库
#include "libresample.h"

// =============================================================================
// 内部结构定义
// =============================================================================

struct AudioProcessor {
    int sample_rate;
    float* temp_buffer;
    size_t temp_buffer_size;
};

// =============================================================================
// 音频处理器
// =============================================================================

AudioProcessor* audio_processor_create(int sample_rate) {
    AudioProcessor* processor = (AudioProcessor*)malloc(sizeof(AudioProcessor));
    if (!processor) {
        return NULL;
    }

    processor->sample_rate = sample_rate;
    processor->temp_buffer_size = 16384;
    processor->temp_buffer = (float*)malloc(processor->temp_buffer_size * sizeof(float));

    if (!processor->temp_buffer) {
        free(processor);
        return NULL;
    }

    LOG_INFO("Audio processor created with sample rate: %d", sample_rate);
    return processor;
}

void audio_processor_destroy(AudioProcessor* processor) {
    if (processor) {
        if (processor->temp_buffer) {
            free(processor->temp_buffer);
        }
        free(processor);
    }
}

// =============================================================================
// 音频缓冲区
// =============================================================================

AudioBuffer audio_buffer_create(size_t capacity) {
    AudioBuffer buffer;
    buffer.capacity = capacity;
    buffer.size = 0;
    buffer.data = (float*)malloc(capacity * sizeof(float));
    buffer.format.sample_rate = 0;
    buffer.format.channels = 1;
    buffer.format.bits_per_sample = 32;
    return buffer;
}

void audio_buffer_free(AudioBuffer* buffer) {
    if (buffer && buffer->data) {
        free(buffer->data);
        buffer->data = NULL;
        buffer->size = 0;
        buffer->capacity = 0;
    }
}

// =============================================================================
// 音频I/O (简单WAV实现)
// =============================================================================

// WAV文件头结构
#pragma pack(push, 1)
typedef struct {
    char riff[4];           // "RIFF"
    uint32_t file_size;     // 文件大小 - 8
    char wave[4];           // "WAVE"
    char fmt[4];            // "fmt "
    uint32_t fmt_size;      // fmt块大小 (16)
    uint16_t audio_format;  // 音频格式 (1=PCM, 3=IEEE float)
    uint16_t num_channels;  // 声道数
    uint32_t sample_rate;   // 采样率
    uint32_t byte_rate;     // 字节率
    uint16_t block_align;   // 块对齐
    uint16_t bits_per_sample; // 位深度
    char data[4];           // "data"
    uint32_t data_size;     // 数据大小
} WAVHeader;
#pragma pack(pop)

int audio_load_file(const char* filepath, AudioBuffer* buffer, AudioFormat* format) {
    if (!filepath || !buffer) {
        return -1;
    }

    FILE* file = fopen(filepath, "rb");
    if (!file) {
        LOG_ERROR("Failed to open file: %s", filepath);
        return -1;
    }

    // 读取WAV头
    WAVHeader header;
    if (fread(&header, sizeof(WAVHeader), 1, file) != 1) {
        LOG_ERROR("Failed to read WAV header");
        fclose(file);
        return -2;
    }

    // 验证WAV格式
    if (memcmp(header.riff, "RIFF", 4) != 0 || memcmp(header.wave, "WAVE", 4) != 0) {
        LOG_ERROR("Invalid WAV file format");
        fclose(file);
        return -3;
    }

    // 跳过可能的额外块，找到data块
    if (memcmp(header.data, "data", 4) != 0) {
        // 需要搜索data块
        fseek(file, 12, SEEK_SET); // 跳过RIFF头
        char chunk_id[4];
        uint32_t chunk_size;

        while (fread(chunk_id, 4, 1, file) == 1) {
            fread(&chunk_size, 4, 1, file);
            if (memcmp(chunk_id, "data", 4) == 0) {
                header.data_size = chunk_size;
                break;
            }
            fseek(file, chunk_size, SEEK_CUR);
        }
    }

    // 设置格式信息
    if (format) {
        format->sample_rate = header.sample_rate;
        format->channels = header.num_channels;
        format->bits_per_sample = header.bits_per_sample;
    }

    // 计算样本数
    size_t num_samples = header.data_size / (header.bits_per_sample / 8) / header.num_channels;

    // 分配缓冲区
    buffer->data = (float*)malloc(num_samples * sizeof(float));
    if (!buffer->data) {
        fclose(file);
        return -4;
    }
    buffer->size = num_samples;
    buffer->capacity = num_samples;
    buffer->format.sample_rate = header.sample_rate;
    buffer->format.channels = header.num_channels;
    buffer->format.bits_per_sample = header.bits_per_sample;

    // 读取音频数据
    if (header.audio_format == 3 && header.bits_per_sample == 32) {
        // IEEE float
        fread(buffer->data, sizeof(float), num_samples, file);
    } else if (header.audio_format == 1 && header.bits_per_sample == 16) {
        // 16-bit PCM
        int16_t* temp = (int16_t*)malloc(num_samples * header.num_channels * sizeof(int16_t));
        fread(temp, sizeof(int16_t), num_samples * header.num_channels, file);

        // 转换为float并混合为单声道
        for (size_t i = 0; i < num_samples; i++) {
            float sum = 0.0f;
            for (int ch = 0; ch < header.num_channels; ch++) {
                sum += temp[i * header.num_channels + ch] / 32768.0f;
            }
            buffer->data[i] = sum / header.num_channels;
        }
        free(temp);
    } else if (header.audio_format == 1 && header.bits_per_sample == 24) {
        // 24-bit PCM
        uint8_t* temp = (uint8_t*)malloc(num_samples * header.num_channels * 3);
        fread(temp, 3, num_samples * header.num_channels, file);

        for (size_t i = 0; i < num_samples; i++) {
            float sum = 0.0f;
            for (int ch = 0; ch < header.num_channels; ch++) {
                size_t idx = (i * header.num_channels + ch) * 3;
                int32_t sample = (temp[idx] | (temp[idx + 1] << 8) | (temp[idx + 2] << 16));
                if (sample & 0x800000) sample |= 0xFF000000; // 符号扩展
                sum += sample / 8388608.0f;
            }
            buffer->data[i] = sum / header.num_channels;
        }
        free(temp);
    } else {
        LOG_ERROR("Unsupported audio format: %d, bits: %d", header.audio_format, header.bits_per_sample);
        free(buffer->data);
        buffer->data = NULL;
        fclose(file);
        return -5;
    }

    fclose(file);
    LOG_INFO("Loaded audio: %zu samples, %d Hz, %d channels",
             buffer->size, header.sample_rate, header.num_channels);
    return 0;
}

int audio_save_file(const char* filepath, const AudioBuffer* buffer, const AudioFormat* format) {
    if (!filepath || !buffer || !buffer->data) {
        return -1;
    }

    FILE* file = fopen(filepath, "wb");
    if (!file) {
        LOG_ERROR("Failed to create file: %s", filepath);
        return -1;
    }

    int sample_rate = format ? format->sample_rate : 16000;
    int channels = format ? format->channels : 1;

    // 准备WAV头
    WAVHeader header;
    memcpy(header.riff, "RIFF", 4);
    memcpy(header.wave, "WAVE", 4);
    memcpy(header.fmt, "fmt ", 4);
    memcpy(header.data, "data", 4);

    header.fmt_size = 16;
    header.audio_format = 3; // IEEE float
    header.num_channels = channels;
    header.sample_rate = sample_rate;
    header.bits_per_sample = 32;
    header.byte_rate = sample_rate * channels * 4;
    header.block_align = channels * 4;
    header.data_size = buffer->size * channels * 4;
    header.file_size = sizeof(WAVHeader) - 8 + header.data_size;

    // 写入头
    fwrite(&header, sizeof(WAVHeader), 1, file);

    // 写入数据
    fwrite(buffer->data, sizeof(float), buffer->size, file);

    fclose(file);
    LOG_INFO("Saved audio: %zu samples to %s", buffer->size, filepath);
    return 0;
}

// =============================================================================
// 音频处理
// =============================================================================

int audio_resample(
    const float* input,
    size_t input_size,
    int src_rate,
    int dst_rate,
    float** output,
    size_t* output_size
) {
    if (!input || !output || !output_size || src_rate <= 0 || dst_rate <= 0) {
        return -1;
    }

    // 如果采样率相同，直接复制
    if (src_rate == dst_rate) {
        *output_size = input_size;
        *output = (float*)malloc(*output_size * sizeof(float));
        if (!*output) {
            return -2;
        }
        memcpy(*output, input, input_size * sizeof(float));
        return 0;
    }

    // 计算重采样因子
    double factor = (double)dst_rate / src_rate;

    // 计算输出大小 (添加一些余量)
    *output_size = (size_t)(input_size * factor + 0.5);

    // 分配输出缓冲区 (多分配一些以防溢出)
    size_t out_buffer_size = *output_size + 1024;
    *output = (float*)malloc(out_buffer_size * sizeof(float));
    if (!*output) {
        return -2;
    }

    // 使用 libresample 高质量重采样
    // highQuality=1 使用 Nmult=35 的 Kaiser 窗口滤波器
    void* resampler = resample_open(1, factor, factor);
    if (!resampler) {
        free(*output);
        *output = NULL;
        LOG_ERROR("Failed to create resampler");
        return -3;
    }

    // 复制输入数据 (resample_process 可能会修改输入)
    float* input_copy = (float*)malloc(input_size * sizeof(float));
    if (!input_copy) {
        resample_close(resampler);
        free(*output);
        *output = NULL;
        return -2;
    }
    memcpy(input_copy, input, input_size * sizeof(float));

    // 执行重采样
    int in_used = 0;
    int out_count = resample_process(
        resampler,
        factor,
        input_copy,
        (int)input_size,
        1,  // lastFlag = 1，表示这是最后一批数据
        &in_used,
        *output,
        (int)out_buffer_size
    );

    free(input_copy);
    resample_close(resampler);

    if (out_count < 0) {
        free(*output);
        *output = NULL;
        LOG_ERROR("Resample failed with error: %d", out_count);
        return -4;
    }

    // 更新实际输出大小
    *output_size = (size_t)out_count;

    LOG_DEBUG("Resampled: %zu -> %zu samples (factor: %.4f)",
              input_size, *output_size, factor);

    return 0;
}

void audio_normalize(float* audio, size_t size, float target_db) {
    if (!audio || size == 0) return;

    // 计算RMS
    double sum_sq = 0.0;
    for (size_t i = 0; i < size; i++) {
        sum_sq += audio[i] * audio[i];
    }
    double rms = sqrt(sum_sq / size);

    if (rms < 1e-8) return;

    // 计算增益
    double target_amp = pow(10.0, target_db / 20.0);
    double gain = target_amp / rms;

    // 应用增益
    for (size_t i = 0; i < size; i++) {
        audio[i] = (float)(audio[i] * gain);
        // 限幅
        if (audio[i] > 1.0f) audio[i] = 1.0f;
        if (audio[i] < -1.0f) audio[i] = -1.0f;
    }
}

int audio_preprocess(
    AudioProcessor* processor,
    const float* input,
    size_t input_size,
    float* output,
    size_t* output_size
) {
    if (!processor || !input || !output || !output_size) {
        return -1;
    }

    // 复制数据
    memcpy(output, input, input_size * sizeof(float));
    *output_size = input_size;

    // 去直流分量
    float mean = array_mean(output, *output_size);
    for (size_t i = 0; i < *output_size; i++) {
        output[i] -= mean;
    }

    // 归一化
    audio_normalize(output, *output_size, -6.0f);

    return 0;
}

int audio_is_silent(const float* audio, size_t size, float threshold_db) {
    if (!audio || size == 0) return 1;

    float rms = audio_rms(audio, size);
    float rms_db = 20.0f * log10f(rms + 1e-8f);

    return rms_db < threshold_db ? 1 : 0;
}

float audio_rms(const float* audio, size_t size) {
    if (!audio || size == 0) return 0.0f;

    double sum_sq = 0.0;
    for (size_t i = 0; i < size; i++) {
        sum_sq += audio[i] * audio[i];
    }
    return (float)sqrt(sum_sq / size);
}

float audio_peak(const float* audio, size_t size) {
    if (!audio || size == 0) return 0.0f;

    float peak = 0.0f;
    for (size_t i = 0; i < size; i++) {
        float abs_val = fabsf(audio[i]);
        if (abs_val > peak) {
            peak = abs_val;
        }
    }
    return peak;
}
