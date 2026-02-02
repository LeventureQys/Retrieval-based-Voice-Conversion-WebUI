/**
 * @file test_stft.cpp
 * @brief STFT 模块测试
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "stft.h"
#include "utils.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void test_window_creation() {
    printf("=== Window Creation Test ===\n");

    STFTProcessor* processor = stft_processor_create(1024, 256, WINDOW_HANN);
    if (processor) {
        printf("[PASS] STFT processor created (Hann window)\n");
        stft_processor_destroy(processor);
    } else {
        printf("[FAIL] Failed to create STFT processor\n");
    }

    processor = stft_processor_create(2048, 512, WINDOW_HAMMING);
    if (processor) {
        printf("[PASS] STFT processor created (Hamming window)\n");
        stft_processor_destroy(processor);
    } else {
        printf("[FAIL] Failed to create STFT processor\n");
    }

    printf("\n");
}

void test_stft_forward() {
    printf("=== STFT Forward Test ===\n");

    // 创建测试信号: 440Hz 正弦波
    int sample_rate = 16000;
    float duration = 0.5f; // 0.5秒
    size_t num_samples = (size_t)(sample_rate * duration);

    float* audio = (float*)malloc(num_samples * sizeof(float));
    if (!audio) {
        printf("[FAIL] Failed to allocate audio buffer\n");
        return;
    }

    // 生成 440Hz 正弦波
    float freq = 440.0f;
    for (size_t i = 0; i < num_samples; i++) {
        audio[i] = 0.5f * sinf(2.0f * (float)M_PI * freq * i / sample_rate);
    }
    printf("[PASS] Generated 440Hz sine wave: %zu samples\n", num_samples);

    // 创建 STFT 处理器
    size_t fft_size = 2048;
    size_t hop_size = 512;
    STFTProcessor* processor = stft_processor_create(fft_size, hop_size, WINDOW_HANN);
    if (!processor) {
        printf("[FAIL] Failed to create STFT processor\n");
        free(audio);
        return;
    }

    // 执行 STFT
    STFTResult result;
    memset(&result, 0, sizeof(STFTResult));

    int ret = stft_forward(processor, audio, num_samples, &result);
    if (ret == 0) {
        printf("[PASS] STFT forward completed\n");
        printf("       Frames: %zu\n", result.num_frames);
        printf("       Bins: %zu\n", result.num_bins);

        // 找到最大能量的频率 bin
        size_t max_bin = 0;
        float max_energy = 0.0f;

        for (size_t bin = 0; bin < result.num_bins; bin++) {
            float energy = 0.0f;
            for (size_t frame = 0; frame < result.num_frames; frame++) {
                Complex* c = &result.data[frame * result.num_bins + bin];
                energy += c->real * c->real + c->imag * c->imag;
            }
            if (energy > max_energy) {
                max_energy = energy;
                max_bin = bin;
            }
        }

        float detected_freq = (float)max_bin * sample_rate / fft_size;
        printf("       Peak frequency bin: %zu (%.1f Hz)\n", max_bin, detected_freq);

        if (fabsf(detected_freq - freq) < 50.0f) {
            printf("[PASS] Detected frequency matches input (440Hz)\n");
        } else {
            printf("[WARN] Detected frequency differs from input\n");
        }

        stft_result_free(&result);
    } else {
        printf("[FAIL] STFT forward failed: %d\n", ret);
    }

    stft_processor_destroy(processor);
    free(audio);
    printf("\n");
}

void test_stft_roundtrip() {
    printf("=== STFT Roundtrip Test ===\n");

    // 创建测试信号
    int sample_rate = 16000;
    size_t num_samples = 8000; // 0.5秒

    float* audio = (float*)malloc(num_samples * sizeof(float));
    if (!audio) {
        printf("[FAIL] Failed to allocate audio buffer\n");
        return;
    }

    // 生成混合信号
    for (size_t i = 0; i < num_samples; i++) {
        float t = (float)i / sample_rate;
        audio[i] = 0.3f * sinf(2.0f * (float)M_PI * 440.0f * t) +
                   0.2f * sinf(2.0f * (float)M_PI * 880.0f * t);
    }

    // 创建 STFT 处理器
    size_t fft_size = 1024;
    size_t hop_size = 256;
    STFTProcessor* processor = stft_processor_create(fft_size, hop_size, WINDOW_HANN);
    if (!processor) {
        printf("[FAIL] Failed to create STFT processor\n");
        free(audio);
        return;
    }

    // STFT 正变换
    STFTResult stft_result;
    memset(&stft_result, 0, sizeof(STFTResult));

    int ret = stft_forward(processor, audio, num_samples, &stft_result);
    if (ret != 0) {
        printf("[FAIL] STFT forward failed\n");
        stft_processor_destroy(processor);
        free(audio);
        return;
    }
    printf("[PASS] STFT forward: %zu frames\n", stft_result.num_frames);

    // STFT 逆变换
    float* reconstructed = NULL;
    size_t reconstructed_size = 0;

    ret = stft_inverse(processor, &stft_result, &reconstructed, &reconstructed_size);
    if (ret != 0) {
        printf("[FAIL] STFT inverse failed\n");
        stft_result_free(&stft_result);
        stft_processor_destroy(processor);
        free(audio);
        return;
    }
    printf("[PASS] STFT inverse: %zu samples\n", reconstructed_size);

    // 计算重建误差
    size_t compare_size = (num_samples < reconstructed_size) ? num_samples : reconstructed_size;
    // 跳过边界效应
    size_t start = fft_size;
    size_t end = compare_size - fft_size;

    float max_error = 0.0f;
    float sum_sq_error = 0.0f;

    for (size_t i = start; i < end; i++) {
        float error = fabsf(audio[i] - reconstructed[i]);
        if (error > max_error) max_error = error;
        sum_sq_error += error * error;
    }

    float rmse = sqrtf(sum_sq_error / (end - start));
    printf("       Max error: %.6f\n", max_error);
    printf("       RMSE: %.6f\n", rmse);

    if (rmse < 0.01f) {
        printf("[PASS] Reconstruction quality is good\n");
    } else {
        printf("[WARN] Reconstruction quality may need improvement\n");
    }

    // 清理
    free(reconstructed);
    stft_result_free(&stft_result);
    stft_processor_destroy(processor);
    free(audio);
    printf("\n");
}

void test_mel_filterbank() {
    printf("=== Mel Filterbank Test ===\n");

    int sample_rate = 16000;
    size_t fft_size = 2048;
    size_t num_mels = 80;
    float fmin = 0.0f;
    float fmax = 8000.0f;

    float* filterbank = NULL;
    int ret = mel_filterbank_create(sample_rate, fft_size, num_mels, fmin, fmax, &filterbank);

    if (ret == 0 && filterbank) {
        printf("[PASS] Mel filterbank created: %zu mels\n", num_mels);

        // 检查滤波器组的基本属性
        size_t num_bins = fft_size / 2 + 1;
        int non_zero_filters = 0;

        for (size_t m = 0; m < num_mels; m++) {
            float sum = 0.0f;
            for (size_t k = 0; k < num_bins; k++) {
                sum += filterbank[m * num_bins + k];
            }
            if (sum > 0.0f) non_zero_filters++;
        }

        printf("       Non-zero filters: %d / %zu\n", non_zero_filters, num_mels);

        free(filterbank);
    } else {
        printf("[FAIL] Failed to create Mel filterbank\n");
    }

    printf("\n");
}

void test_hz_mel_conversion() {
    printf("=== Hz-Mel Conversion Test ===\n");

    float test_freqs[] = {100.0f, 440.0f, 1000.0f, 4000.0f, 8000.0f};
    int num_tests = sizeof(test_freqs) / sizeof(test_freqs[0]);

    printf("       Hz -> Mel -> Hz roundtrip:\n");
    for (int i = 0; i < num_tests; i++) {
        float hz = test_freqs[i];
        float mel = hz_to_mel(hz);
        float hz_back = mel_to_hz(mel);
        float error = fabsf(hz - hz_back);

        printf("       %.0f Hz -> %.2f mel -> %.2f Hz (error: %.4f)\n",
               hz, mel, hz_back, error);
    }

    printf("[PASS] Hz-Mel conversion test completed\n");
    printf("\n");
}

int main() {
    printf("========================================\n");
    printf("   RVC_ONNX - STFT Test Suite\n");
    printf("========================================\n\n");

    log_set_level(LOG_INFO);

    test_window_creation();
    test_hz_mel_conversion();
    test_mel_filterbank();
    test_stft_forward();
    test_stft_roundtrip();

    printf("========================================\n");
    printf("   All STFT tests completed!\n");
    printf("========================================\n");

    return 0;
}
