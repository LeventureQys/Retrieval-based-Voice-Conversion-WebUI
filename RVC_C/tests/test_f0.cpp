/**
 * @file test_f0.cpp
 * @brief F0 提取模块测试
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "f0_extractor.h"
#include "utils.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void test_f0_extractor_creation() {
    printf("=== F0 Extractor Creation Test ===\n");

    F0Extractor* extractor = f0_extractor_create(F0_METHOD_HARVEST, 16000);
    if (extractor) {
        printf("[PASS] Harvest F0 extractor created\n");
        f0_extractor_destroy(extractor);
    } else {
        printf("[FAIL] Failed to create Harvest extractor\n");
    }

    extractor = f0_extractor_create(F0_METHOD_DIO, 16000);
    if (extractor) {
        printf("[PASS] DIO F0 extractor created\n");
        f0_extractor_destroy(extractor);
    } else {
        printf("[FAIL] Failed to create DIO extractor\n");
    }

    printf("\n");
}

void test_f0_extraction_sine() {
    printf("=== F0 Extraction Test (Sine Wave) ===\n");

    // 创建测试信号: 纯正弦波
    int sample_rate = 16000;
    float duration = 1.0f;
    size_t num_samples = (size_t)(sample_rate * duration);

    float* audio = (float*)malloc(num_samples * sizeof(float));
    if (!audio) {
        printf("[FAIL] Failed to allocate audio buffer\n");
        return;
    }

    // 生成 220Hz 正弦波 (A3)
    float target_freq = 220.0f;
    for (size_t i = 0; i < num_samples; i++) {
        audio[i] = 0.8f * sinf(2.0f * (float)M_PI * target_freq * i / sample_rate);
    }
    printf("[INFO] Generated %.0f Hz sine wave: %zu samples\n", target_freq, num_samples);

    // 测试 Harvest
    F0Extractor* extractor = f0_extractor_create(F0_METHOD_HARVEST, sample_rate);
    if (!extractor) {
        printf("[FAIL] Failed to create extractor\n");
        free(audio);
        return;
    }

    F0Result result;
    memset(&result, 0, sizeof(F0Result));

    int64_t start_time = get_time_ms();
    int ret = f0_extract_float(extractor, audio, num_samples, &result);
    int64_t elapsed = get_time_ms() - start_time;

    if (ret == 0) {
        printf("[PASS] Harvest extraction completed in %lld ms\n", (long long)elapsed);
        printf("       F0 frames: %zu\n", result.length);

        // 计算平均 F0 (排除静音帧)
        double sum_f0 = 0.0;
        int voiced_count = 0;
        for (size_t i = 0; i < result.length; i++) {
            if (result.f0[i] > 0) {
                sum_f0 += result.f0[i];
                voiced_count++;
            }
        }

        if (voiced_count > 0) {
            double avg_f0 = sum_f0 / voiced_count;
            double error = fabs(avg_f0 - target_freq);
            printf("       Average F0: %.2f Hz (target: %.0f Hz)\n", avg_f0, target_freq);
            printf("       Error: %.2f Hz (%.1f%%)\n", error, error / target_freq * 100);

            if (error < 10.0) {
                printf("[PASS] F0 detection accuracy is good\n");
            } else {
                printf("[WARN] F0 detection error is high\n");
            }
        } else {
            printf("[WARN] No voiced frames detected\n");
        }

        f0_result_free(&result);
    } else {
        printf("[FAIL] F0 extraction failed: %d\n", ret);
    }

    f0_extractor_destroy(extractor);
    free(audio);
    printf("\n");
}

void test_f0_extraction_sweep() {
    printf("=== F0 Extraction Test (Frequency Sweep) ===\n");

    // 创建频率扫描信号
    int sample_rate = 16000;
    float duration = 2.0f;
    size_t num_samples = (size_t)(sample_rate * duration);

    float* audio = (float*)malloc(num_samples * sizeof(float));
    if (!audio) {
        printf("[FAIL] Failed to allocate audio buffer\n");
        return;
    }

    // 生成 100Hz 到 400Hz 的频率扫描
    float start_freq = 100.0f;
    float end_freq = 400.0f;

    for (size_t i = 0; i < num_samples; i++) {
        float t = (float)i / sample_rate;
        float freq = start_freq + (end_freq - start_freq) * t / duration;
        float phase = 2.0f * (float)M_PI * (start_freq * t + 0.5f * (end_freq - start_freq) * t * t / duration);
        audio[i] = 0.8f * sinf(phase);
    }
    printf("[INFO] Generated frequency sweep: %.0f Hz -> %.0f Hz\n", start_freq, end_freq);

    // 测试 DIO
    F0Extractor* extractor = f0_extractor_create(F0_METHOD_DIO, sample_rate);
    if (!extractor) {
        printf("[FAIL] Failed to create extractor\n");
        free(audio);
        return;
    }

    F0Result result;
    memset(&result, 0, sizeof(F0Result));

    int64_t start_time = get_time_ms();
    int ret = f0_extract_float(extractor, audio, num_samples, &result);
    int64_t elapsed = get_time_ms() - start_time;

    if (ret == 0) {
        printf("[PASS] DIO extraction completed in %lld ms\n", (long long)elapsed);
        printf("       F0 frames: %zu\n", result.length);

        // 检查 F0 是否单调递增
        int monotonic = 1;
        double prev_f0 = 0;
        int voiced_count = 0;

        for (size_t i = 0; i < result.length; i++) {
            if (result.f0[i] > 0) {
                if (prev_f0 > 0 && result.f0[i] < prev_f0 - 20) {
                    monotonic = 0;
                }
                prev_f0 = result.f0[i];
                voiced_count++;
            }
        }

        printf("       Voiced frames: %d / %zu\n", voiced_count, result.length);

        if (voiced_count > 0) {
            // 打印首尾 F0
            double first_f0 = 0, last_f0 = 0;
            for (size_t i = 0; i < result.length; i++) {
                if (result.f0[i] > 0) {
                    first_f0 = result.f0[i];
                    break;
                }
            }
            for (size_t i = result.length; i > 0; i--) {
                if (result.f0[i-1] > 0) {
                    last_f0 = result.f0[i-1];
                    break;
                }
            }
            printf("       First F0: %.2f Hz, Last F0: %.2f Hz\n", first_f0, last_f0);

            if (monotonic) {
                printf("[PASS] F0 trend is monotonically increasing\n");
            } else {
                printf("[WARN] F0 trend is not strictly monotonic\n");
            }
        }

        f0_result_free(&result);
    } else {
        printf("[FAIL] F0 extraction failed: %d\n", ret);
    }

    f0_extractor_destroy(extractor);
    free(audio);
    printf("\n");
}

void test_f0_postprocessing() {
    printf("=== F0 Post-processing Test ===\n");

    // 创建带噪声的 F0 序列
    size_t length = 100;
    double* f0 = (double*)malloc(length * sizeof(double));
    if (!f0) {
        printf("[FAIL] Failed to allocate F0 buffer\n");
        return;
    }

    // 生成带噪声和间断的 F0
    double base_f0 = 200.0;
    for (size_t i = 0; i < length; i++) {
        if (i >= 30 && i < 40) {
            f0[i] = 0; // 模拟无声段
        } else {
            f0[i] = base_f0 + 10.0 * sin(i * 0.1) + (rand() % 20 - 10);
        }
    }

    printf("[INFO] Created F0 sequence with gaps and noise\n");

    // 测试插值
    double* f0_interp = (double*)malloc(length * sizeof(double));
    memcpy(f0_interp, f0, length * sizeof(double));
    f0_interpolate(f0_interp, length);

    int gaps_filled = 1;
    for (size_t i = 0; i < length; i++) {
        if (f0_interp[i] <= 0) {
            gaps_filled = 0;
            break;
        }
    }

    if (gaps_filled) {
        printf("[PASS] F0 interpolation filled all gaps\n");
    } else {
        printf("[WARN] F0 interpolation did not fill all gaps\n");
    }

    // 测试平滑
    double* f0_smooth_arr = (double*)malloc(length * sizeof(double));
    memcpy(f0_smooth_arr, f0_interp, length * sizeof(double));
    f0_smooth(f0_smooth_arr, length, 5);

    // 计算平滑前后的标准差
    double sum1 = 0, sum2 = 0;
    for (size_t i = 0; i < length; i++) {
        sum1 += f0_interp[i];
        sum2 += f0_smooth_arr[i];
    }
    double mean1 = sum1 / length;
    double mean2 = sum2 / length;

    double var1 = 0, var2 = 0;
    for (size_t i = 0; i < length; i++) {
        var1 += (f0_interp[i] - mean1) * (f0_interp[i] - mean1);
        var2 += (f0_smooth_arr[i] - mean2) * (f0_smooth_arr[i] - mean2);
    }
    double std1 = sqrt(var1 / length);
    double std2 = sqrt(var2 / length);

    printf("       Std before smoothing: %.2f\n", std1);
    printf("       Std after smoothing: %.2f\n", std2);

    if (std2 < std1) {
        printf("[PASS] Smoothing reduced variance\n");
    } else {
        printf("[WARN] Smoothing did not reduce variance\n");
    }

    // 测试音高偏移
    double* f0_shifted = (double*)malloc(length * sizeof(double));
    memcpy(f0_shifted, f0_smooth_arr, length * sizeof(double));
    f0_shift_pitch(f0_shifted, length, 12.0); // 上移一个八度

    double ratio = f0_shifted[50] / f0_smooth_arr[50];
    printf("       Pitch shift ratio (12 semitones): %.4f (expected: 2.0)\n", ratio);

    if (fabs(ratio - 2.0) < 0.01) {
        printf("[PASS] Pitch shift is correct\n");
    } else {
        printf("[WARN] Pitch shift ratio is incorrect\n");
    }

    free(f0);
    free(f0_interp);
    free(f0_smooth_arr);
    free(f0_shifted);
    printf("\n");
}

void test_freq_midi_conversion() {
    printf("=== Frequency-MIDI Conversion Test ===\n");

    // 测试已知的频率-MIDI对应关系
    struct {
        double freq;
        double midi;
        const char* note;
    } test_cases[] = {
        {440.0, 69.0, "A4"},
        {261.63, 60.0, "C4"},
        {880.0, 81.0, "A5"},
        {220.0, 57.0, "A3"},
    };

    int num_tests = sizeof(test_cases) / sizeof(test_cases[0]);
    int passed = 0;

    for (int i = 0; i < num_tests; i++) {
        double midi = freq_to_midi(test_cases[i].freq);
        double freq_back = midi_to_freq(test_cases[i].midi);

        double midi_error = fabs(midi - test_cases[i].midi);
        double freq_error = fabs(freq_back - test_cases[i].freq);

        printf("       %s: %.2f Hz -> MIDI %.2f (expected %.0f), error: %.4f\n",
               test_cases[i].note, test_cases[i].freq, midi, test_cases[i].midi, midi_error);

        if (midi_error < 0.1 && freq_error < 0.1) {
            passed++;
        }
    }

    if (passed == num_tests) {
        printf("[PASS] All frequency-MIDI conversions correct\n");
    } else {
        printf("[WARN] %d/%d conversions correct\n", passed, num_tests);
    }

    printf("\n");
}

int main() {
    printf("========================================\n");
    printf("   RVC_ONNX - F0 Extraction Test Suite\n");
    printf("========================================\n\n");

    log_set_level(LOG_INFO);

    test_f0_extractor_creation();
    test_freq_midi_conversion();
    test_f0_postprocessing();
    test_f0_extraction_sine();
    test_f0_extraction_sweep();

    printf("========================================\n");
    printf("   All F0 tests completed!\n");
    printf("========================================\n");

    return 0;
}
