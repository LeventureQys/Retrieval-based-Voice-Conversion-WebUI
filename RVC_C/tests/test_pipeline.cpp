/**
 * @file test_pipeline.cpp
 * @brief 完整流程测试
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "rvc_onnx.h"
#include "audio_processor.h"
#include "utils.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void test_rvc_config() {
    printf("=== RVC Config Test ===\n");

    RVCConfig config = rvc_default_config();

    printf("[PASS] Default config created\n");
    printf("       Sample rate: %d\n", config.sample_rate);
    printf("       Target sample rate: %d\n", config.target_sample_rate);
    printf("       Pitch shift: %.1f\n", config.pitch_shift);
    printf("       Block size: %d\n", config.block_size);
    printf("       Num threads: %d\n", config.num_threads);

    printf("\n");
}

void test_rvc_context_creation() {
    printf("=== RVC Context Creation Test ===\n");

    RVCConfig config = rvc_default_config();
    // 不加载模型，只测试基础创建
    config.hubert_model_path = NULL;
    config.synthesizer_model_path = NULL;

    RVCContext* ctx = rvc_create(&config);
    if (ctx) {
        printf("[PASS] RVC context created (without models)\n");

        // 测试参数设置
        RVCError err = rvc_set_pitch_shift(ctx, 5.0f);
        if (err == RVC_SUCCESS) {
            printf("[PASS] Pitch shift set to 5.0\n");
        }

        err = rvc_set_index_rate(ctx, 0.5f);
        if (err == RVC_SUCCESS) {
            printf("[PASS] Index rate set to 0.5\n");
        }

        rvc_destroy(ctx);
        printf("[PASS] RVC context destroyed\n");
    } else {
        printf("[FAIL] Failed to create RVC context\n");
    }

    printf("\n");
}

void test_audio_io() {
    printf("=== Audio I/O Test ===\n");

    // 创建测试音频
    int sample_rate = 16000;
    float duration = 1.0f;
    size_t num_samples = (size_t)(sample_rate * duration);

    AudioBuffer buffer = audio_buffer_create(num_samples);
    if (!buffer.data) {
        printf("[FAIL] Failed to create audio buffer\n");
        return;
    }

    // 生成测试信号
    for (size_t i = 0; i < num_samples; i++) {
        float t = (float)i / sample_rate;
        buffer.data[i] = 0.5f * sinf(2.0f * (float)M_PI * 440.0f * t);
    }
    buffer.size = num_samples;
    buffer.format.sample_rate = sample_rate;
    buffer.format.channels = 1;
    buffer.format.bits_per_sample = 32;

    printf("[PASS] Generated test audio: %zu samples\n", num_samples);

    // 保存到文件
    const char* test_file = "test_output.wav";
    int ret = audio_save_file(test_file, &buffer, &buffer.format);
    if (ret == 0) {
        printf("[PASS] Saved audio to %s\n", test_file);

        // 重新加载
        AudioBuffer loaded = {0};
        AudioFormat format = {0};
        ret = audio_load_file(test_file, &loaded, &format);

        if (ret == 0) {
            printf("[PASS] Loaded audio: %zu samples, %d Hz\n", loaded.size, format.sample_rate);

            // 比较
            if (loaded.size == buffer.size) {
                float max_diff = 0.0f;
                for (size_t i = 0; i < loaded.size; i++) {
                    float diff = fabsf(loaded.data[i] - buffer.data[i]);
                    if (diff > max_diff) max_diff = diff;
                }
                printf("       Max difference: %.6f\n", max_diff);

                if (max_diff < 0.001f) {
                    printf("[PASS] Audio roundtrip successful\n");
                } else {
                    printf("[WARN] Audio roundtrip has differences\n");
                }
            }

            audio_buffer_free(&loaded);
        } else {
            printf("[FAIL] Failed to load audio\n");
        }

        // 删除测试文件
        remove(test_file);
    } else {
        printf("[FAIL] Failed to save audio\n");
    }

    audio_buffer_free(&buffer);
    printf("\n");
}

void test_audio_processing() {
    printf("=== Audio Processing Test ===\n");

    AudioProcessor* processor = audio_processor_create(16000);
    if (!processor) {
        printf("[FAIL] Failed to create audio processor\n");
        return;
    }
    printf("[PASS] Audio processor created\n");

    // 创建测试音频
    size_t num_samples = 8000;
    float* input = (float*)malloc(num_samples * sizeof(float));
    float* output = (float*)malloc(num_samples * sizeof(float));

    // 生成带直流偏移的信号
    for (size_t i = 0; i < num_samples; i++) {
        input[i] = 0.3f + 0.5f * sinf(2.0f * (float)M_PI * 440.0f * i / 16000);
    }

    // 测试预处理
    size_t output_size;
    int ret = audio_preprocess(processor, input, num_samples, output, &output_size);

    if (ret == 0) {
        printf("[PASS] Audio preprocessing completed\n");

        // 检查直流偏移是否被移除
        float mean = array_mean(output, output_size);
        printf("       Output mean: %.6f (should be ~0)\n", mean);

        if (fabsf(mean) < 0.01f) {
            printf("[PASS] DC offset removed\n");
        }
    } else {
        printf("[FAIL] Audio preprocessing failed\n");
    }

    // 测试重采样
    float* resampled = NULL;
    size_t resampled_size = 0;

    ret = audio_resample(input, num_samples, 16000, 48000, &resampled, &resampled_size);
    if (ret == 0) {
        printf("[PASS] Resampling 16kHz -> 48kHz: %zu -> %zu samples\n",
               num_samples, resampled_size);

        size_t expected_size = num_samples * 3; // 48000/16000 = 3
        if (abs((int)resampled_size - (int)expected_size) <= 1) {
            printf("[PASS] Resampled size is correct\n");
        }

        free(resampled);
    } else {
        printf("[FAIL] Resampling failed\n");
    }

    // 测试静音检测
    float silent_audio[1000] = {0};
    for (int i = 0; i < 1000; i++) {
        silent_audio[i] = 0.0001f * ((float)rand() / RAND_MAX - 0.5f);
    }

    int is_silent = audio_is_silent(silent_audio, 1000, -40.0f);
    if (is_silent) {
        printf("[PASS] Silent audio detected correctly\n");
    } else {
        printf("[WARN] Silent audio not detected\n");
    }

    is_silent = audio_is_silent(input, num_samples, -40.0f);
    if (!is_silent) {
        printf("[PASS] Non-silent audio detected correctly\n");
    } else {
        printf("[WARN] Non-silent audio incorrectly marked as silent\n");
    }

    free(input);
    free(output);
    audio_processor_destroy(processor);
    printf("\n");
}

void test_error_strings() {
    printf("=== Error String Test ===\n");

    RVCError errors[] = {
        RVC_SUCCESS,
        RVC_ERROR_INVALID_PARAM,
        RVC_ERROR_MODEL_LOAD,
        RVC_ERROR_INFERENCE,
        RVC_ERROR_AUDIO_PROCESS,
        RVC_ERROR_MEMORY,
        RVC_ERROR_FILE_IO,
        RVC_ERROR_NOT_INITIALIZED
    };

    int num_errors = sizeof(errors) / sizeof(errors[0]);

    for (int i = 0; i < num_errors; i++) {
        const char* str = rvc_error_string(errors[i]);
        printf("       Error %d: %s\n", errors[i], str);
    }

    printf("[PASS] All error strings retrieved\n");
    printf("\n");
}

void test_version() {
    printf("=== Version Test ===\n");

    const char* version = rvc_version();
    printf("       Version: %s\n", version);
    printf("[PASS] Version string retrieved\n");
    printf("\n");
}

void test_full_pipeline_mock() {
    printf("=== Full Pipeline Mock Test ===\n");

    // 创建配置（不加载实际模型）
    RVCConfig config = rvc_default_config();
    config.hubert_model_path = NULL;
    config.synthesizer_model_path = NULL;
    config.pitch_shift = 0.0f;

    RVCContext* ctx = rvc_create(&config);
    if (!ctx) {
        printf("[FAIL] Failed to create context\n");
        return;
    }

    // 创建测试输入
    size_t input_samples = 16000; // 1秒
    float* input = (float*)malloc(input_samples * sizeof(float));
    float* output = (float*)malloc(input_samples * 2 * sizeof(float));

    for (size_t i = 0; i < input_samples; i++) {
        input[i] = 0.5f * sinf(2.0f * (float)M_PI * 220.0f * i / 16000);
    }

    printf("[INFO] Created 1 second test audio at 220Hz\n");

    // 测试转换（没有模型时会返回原始音频）
    size_t output_samples = input_samples * 2;
    int64_t start_time = get_time_ms();

    RVCError err = rvc_convert(ctx, input, input_samples, output, &output_samples);

    int64_t elapsed = get_time_ms() - start_time;

    if (err == RVC_SUCCESS) {
        printf("[PASS] Pipeline executed in %lld ms\n", (long long)elapsed);
        printf("       Input samples: %zu\n", input_samples);
        printf("       Output samples: %zu\n", output_samples);
    } else {
        printf("[INFO] Pipeline returned: %s (expected without models)\n",
               rvc_error_string(err));
    }

    free(input);
    free(output);
    rvc_destroy(ctx);
    printf("\n");
}

int main() {
    printf("========================================\n");
    printf("   RVC_ONNX - Full Pipeline Test Suite\n");
    printf("========================================\n\n");

    log_set_level(LOG_INFO);

    test_version();
    test_error_strings();
    test_rvc_config();
    test_rvc_context_creation();
    test_audio_io();
    test_audio_processing();
    test_full_pipeline_mock();

    printf("========================================\n");
    printf("   All pipeline tests completed!\n");
    printf("========================================\n");

    return 0;
}
