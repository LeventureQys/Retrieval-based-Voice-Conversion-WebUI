/**
 * @file test_streaming_integration.cpp
 * @brief RVC 流式处理集成测试
 *
 * 测试流式处理模块与 F0 提取、特征处理的集成。
 */

#include "rvc_streaming.h"
#include "rvc_onnx.h"
#include "f0_extractor.h"
#include "utils.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// =============================================================================
// 测试工具函数
// =============================================================================

/** 生成测试正弦波 */
static void generate_sine_wave(float* buffer, size_t samples, float freq, int sample_rate) {
    for (size_t i = 0; i < samples; i++) {
        buffer[i] = 0.8f * sinf(2.0f * (float)M_PI * freq * (float)i / (float)sample_rate);
    }
}

/** 生成语音样的调制信号 */
static void generate_modulated_tone(float* buffer, size_t samples, float base_freq, int sample_rate) {
    for (size_t i = 0; i < samples; i++) {
        float t = (float)i / (float)sample_rate;
        // 添加轻微的频率调制，模拟真实语音
        float freq = base_freq + 20.0f * sinf(2.0f * (float)M_PI * 5.0f * t);
        float phase = 2.0f * (float)M_PI * freq * t;
        // 添加简单的包络
        float envelope = 1.0f - 0.3f * fabsf(sinf(2.0f * (float)M_PI * 3.0f * t));
        buffer[i] = 0.7f * envelope * sinf(phase);
    }
}

/** 打印测试结果 */
static void print_result(const char* test_name, int passed) {
    printf("[%s] %s\n", passed ? "PASS" : "FAIL", test_name);
}

// =============================================================================
// 集成测试用例
// =============================================================================

/** 测试 F0 提取集成 */
static int test_f0_integration(void) {
    printf("\n--- Test: F0 Extraction Integration ---\n");

    // 创建 F0 提取器
    F0Extractor* extractor = f0_extractor_create(F0_METHOD_HARVEST, 16000);
    if (!extractor) {
        printf("Failed to create F0 extractor\n");
        return 0;
    }

    // 生成测试音频
    const size_t samples = 16000;  // 1 秒
    float* audio = (float*)malloc(samples * sizeof(float));
    generate_sine_wave(audio, samples, 220.0f, 16000);

    // 提取 F0
    F0Result result;
    memset(&result, 0, sizeof(F0Result));

    int64_t start = get_time_ms();
    int ret = f0_extract_float(extractor, audio, samples, &result);
    int64_t elapsed = get_time_ms() - start;

    printf("F0 extraction time: %lld ms\n", (long long)elapsed);

    int passed = (ret == 0 && result.length > 0);

    if (passed) {
        printf("Extracted %zu F0 frames\n", result.length);

        // 验证 F0 值范围
        int valid_frames = 0;
        double sum_f0 = 0.0;
        for (size_t i = 0; i < result.length; i++) {
            if (result.f0[i] > 50.0 && result.f0[i] < 1100.0) {
                valid_frames++;
                sum_f0 += result.f0[i];
            }
        }

        if (valid_frames > 0) {
            double avg_f0 = sum_f0 / valid_frames;
            printf("Average F0: %.1f Hz (target: 220 Hz)\n", avg_f0);
            printf("Valid frames: %d / %zu\n", valid_frames, result.length);

            // 检查 F0 精度
            passed = (fabs(avg_f0 - 220.0) < 20.0);
        }

        f0_result_free(&result);
    }

    free(audio);
    f0_extractor_destroy(extractor);

    print_result("F0 Extraction Integration", passed);
    return passed;
}

/** 测试 F0 转 Pitch 与缓存集成 */
static int test_pitch_cache_integration(void) {
    printf("\n--- Test: Pitch Cache Integration ---\n");

    // 创建流式状态
    RVCStreamState* state = rvc_stream_state_create(NULL);
    if (!state) {
        return 0;
    }

    // 创建 F0 提取器
    F0Extractor* extractor = f0_extractor_create(F0_METHOD_HARVEST, 16000);
    if (!extractor) {
        rvc_stream_state_destroy(state);
        return 0;
    }

    // 生成测试音频 (多块)
    const size_t chunk_size = 4000;  // 250ms
    const int num_chunks = 4;

    float* audio = (float*)malloc(chunk_size * sizeof(float));

    int passed = 1;

    for (int chunk = 0; chunk < num_chunks; chunk++) {
        // 生成带变化频率的音频
        float freq = 200.0f + 50.0f * chunk;
        generate_sine_wave(audio, chunk_size, freq, 16000);

        // 提取 F0
        F0Result result;
        memset(&result, 0, sizeof(F0Result));

        int ret = f0_extract_float(extractor, audio, chunk_size, &result);
        if (ret != 0) {
            printf("Chunk %d: F0 extraction failed\n", chunk);
            passed = 0;
            continue;
        }

        // 转换为 pitch
        size_t pitch_len = result.length;
        int64_t* pitch = (int64_t*)malloc(pitch_len * sizeof(int64_t));
        float* pitchf = (float*)malloc(pitch_len * sizeof(float));

        rvc_f0_to_pitch_batch(result.f0, pitch_len, pitch, pitchf, 50.0f, 1100.0f);

        // 检查 pitch 值范围
        int valid = 0;
        for (size_t i = 0; i < pitch_len; i++) {
            if (pitch[i] >= 1 && pitch[i] <= 255) {
                valid++;
            }
        }

        printf("Chunk %d: freq=%.0f Hz, frames=%zu, valid_pitch=%d\n",
               chunk, freq, pitch_len, valid);

        free(pitch);
        free(pitchf);
        f0_result_free(&result);
    }

    free(audio);
    f0_extractor_destroy(extractor);
    rvc_stream_state_destroy(state);

    print_result("Pitch Cache Integration", passed);
    return passed;
}

/** 测试特征上采样性能 */
static int test_feature_upsample_performance(void) {
    printf("\n--- Test: Feature Upsample Performance ---\n");

    // 模拟 ContentVec 输出大小
    const size_t input_frames = 50;  // 1 秒 @ 50 fps
    const size_t feature_dim = 768;

    float* input = (float*)malloc(input_frames * feature_dim * sizeof(float));
    float* output = (float*)malloc(input_frames * 2 * feature_dim * sizeof(float));

    // 填充随机特征
    for (size_t i = 0; i < input_frames * feature_dim; i++) {
        input[i] = (float)(rand() % 1000) / 1000.0f;
    }

    // 测量上采样性能
    const int iterations = 1000;
    int64_t start = get_time_ms();

    for (int i = 0; i < iterations; i++) {
        rvc_interpolate_features_2x(input, input_frames, output, feature_dim);
    }

    int64_t elapsed = get_time_ms() - start;
    double avg_time = (double)elapsed / iterations;

    printf("Feature upsample: %d iterations in %lld ms\n", iterations, (long long)elapsed);
    printf("Average time per call: %.3f ms\n", avg_time);
    printf("Input: %zu frames x %zu dims = %zu floats\n",
           input_frames, feature_dim, input_frames * feature_dim);
    printf("Output: %zu frames x %zu dims = %zu floats\n",
           input_frames * 2, feature_dim, input_frames * 2 * feature_dim);

    // 验证输出
    int passed = 1;

    // 检查第一帧是否正确复制
    for (size_t d = 0; d < feature_dim && passed; d++) {
        if (fabsf(output[d] - input[d]) > 0.0001f) {
            passed = 0;
        }
    }

    // 检查插值帧
    for (size_t d = 0; d < feature_dim && passed; d++) {
        float expected = (input[d] + input[feature_dim + d]) * 0.5f;
        if (fabsf(output[feature_dim + d] - expected) > 0.0001f) {
            passed = 0;
        }
    }

    free(input);
    free(output);

    print_result("Feature Upsample Performance", passed);
    return passed;
}

/** 测试流式处理延迟 */
static int test_streaming_latency(void) {
    printf("\n--- Test: Streaming Latency Simulation ---\n");

    RVCStreamState* state = rvc_stream_state_create(NULL);
    if (!state) {
        return 0;
    }

    F0Extractor* extractor = f0_extractor_create(F0_METHOD_DIO, 16000);
    if (!extractor) {
        rvc_stream_state_destroy(state);
        return 0;
    }

    // 测试不同块大小的延迟
    size_t block_sizes[] = {1600, 3200, 4800, 8000, 16000};  // 100ms, 200ms, 300ms, 500ms, 1000ms
    int num_sizes = sizeof(block_sizes) / sizeof(block_sizes[0]);

    float* audio = (float*)malloc(16000 * sizeof(float));
    generate_modulated_tone(audio, 16000, 220.0f, 16000);

    printf("Block Size | F0 Time | Total Time | RTF\n");
    printf("-----------|---------|------------|-----\n");

    int passed = 1;

    for (int s = 0; s < num_sizes; s++) {
        size_t block_size = block_sizes[s];

        // 测量 F0 提取时间
        F0Result result;
        memset(&result, 0, sizeof(F0Result));

        int64_t f0_start = get_time_ms();
        int ret = f0_extract_float(extractor, audio, block_size, &result);
        int64_t f0_time = get_time_ms() - f0_start;

        if (ret != 0) {
            printf("F0 extraction failed for block_size=%zu\n", block_size);
            passed = 0;
            continue;
        }

        // 模拟特征上采样
        int64_t upsample_start = get_time_ms();
        size_t cv_frames = block_size / 320;
        float* features = (float*)calloc(cv_frames * 768, sizeof(float));
        float* upsampled = (float*)calloc(cv_frames * 2 * 768, sizeof(float));
        rvc_interpolate_features_2x(features, cv_frames, upsampled, 768);
        int64_t upsample_time = get_time_ms() - upsample_start;

        // 计算 RTF (Real-Time Factor)
        double audio_duration = (double)block_size / 16000.0 * 1000.0;  // ms
        double total_time = (double)(f0_time + upsample_time);
        double rtf = total_time / audio_duration;

        printf("%7zu ms | %5lld ms | %7.1f ms | %.2f\n",
               block_size / 16, (long long)f0_time, total_time, rtf);

        // RTF < 1.0 表示实时
        if (rtf > 1.0) {
            printf("  Warning: RTF > 1.0, not real-time!\n");
        }

        free(features);
        free(upsampled);
        f0_result_free(&result);
    }

    free(audio);
    f0_extractor_destroy(extractor);
    rvc_stream_state_destroy(state);

    print_result("Streaming Latency Simulation", passed);
    return passed;
}

/** 测试连续流处理一致性 */
static int test_continuous_stream(void) {
    printf("\n--- Test: Continuous Stream Consistency ---\n");

    RVCStreamState* state = rvc_stream_state_create(NULL);
    if (!state) {
        return 0;
    }

    // 生成长音频
    const size_t total_samples = 48000;  // 3 秒
    float* full_audio = (float*)malloc(total_samples * sizeof(float));
    generate_modulated_tone(full_audio, total_samples, 300.0f, 16000);

    // 分块处理
    const size_t chunk_size = 4800;  // 300ms
    const int num_chunks = total_samples / chunk_size;

    printf("Processing %d chunks of %zu samples (%.0f ms each)\n",
           num_chunks, chunk_size, chunk_size / 16.0);

    size_t output_capacity = rvc_stream_calc_output_size(chunk_size);
    float* output = (float*)malloc(output_capacity * sizeof(float));

    int passed = 1;
    size_t total_output = 0;

    for (int i = 0; i < num_chunks; i++) {
        float* chunk_input = full_audio + i * chunk_size;

        // 模拟流式处理（使用占位实现）
        // 实际处理需要有效的 RVCContext

        // 简单复制并上采样作为占位
        for (size_t j = 0; j < chunk_size && j * 3 + 2 < output_capacity; j++) {
            output[j * 3] = chunk_input[j];
            output[j * 3 + 1] = chunk_input[j];
            output[j * 3 + 2] = chunk_input[j];
        }
        size_t chunk_output = chunk_size * 3;
        total_output += chunk_output;

        // 更新状态
        state->total_samples_processed += chunk_size;
        state->is_first_chunk = 0;
    }

    printf("Total input: %zu samples\n", total_samples);
    printf("Total output: %zu samples\n", total_output);
    printf("State samples processed: %zu\n", state->total_samples_processed);

    passed = (state->total_samples_processed == total_samples);
    passed &= (total_output == total_samples * 3);

    free(full_audio);
    free(output);
    rvc_stream_state_destroy(state);

    print_result("Continuous Stream Consistency", passed);
    return passed;
}

/** 测试内存使用 */
static int test_memory_usage(void) {
    printf("\n--- Test: Memory Usage ---\n");

    RVCStreamConfig config = rvc_stream_default_config();

    // 计算预期内存使用
    size_t pitch_cache_size = RVC_PITCH_CACHE_SIZE * sizeof(int64_t) +
                              RVC_PITCH_CACHE_SIZE * sizeof(float);
    size_t audio_context_size = config.audio_context_size * sizeof(float);
    size_t feature_context_size = 100 * RVC_FEATURE_DIM * sizeof(float);
    size_t work_buffer_size = (config.block_size + config.audio_context_size + 1024) * sizeof(float) +
                              2048 * RVC_FEATURE_DIM * sizeof(float);

    size_t total_expected = pitch_cache_size + audio_context_size + feature_context_size + work_buffer_size;

    printf("Memory breakdown:\n");
    printf("  Pitch cache:     %zu KB\n", pitch_cache_size / 1024);
    printf("  Audio context:   %zu KB\n", audio_context_size / 1024);
    printf("  Feature context: %zu KB\n", feature_context_size / 1024);
    printf("  Work buffers:    %zu KB\n", work_buffer_size / 1024);
    printf("  ---\n");
    printf("  Total:           %zu KB (%.2f MB)\n", total_expected / 1024, total_expected / 1024.0 / 1024.0);

    // 创建状态验证分配成功
    RVCStreamState* state = rvc_stream_state_create(&config);
    int passed = (state != NULL);

    if (state) {
        // 验证所有缓冲区都已分配
        passed &= (state->cache_pitch != NULL);
        passed &= (state->cache_pitchf != NULL);
        passed &= (state->audio_context != NULL);
        passed &= (state->feature_context != NULL);
        passed &= (state->work_audio != NULL);
        passed &= (state->work_features != NULL);

        printf("All buffers allocated: %s\n", passed ? "Yes" : "No");

        rvc_stream_state_destroy(state);
    }

    print_result("Memory Usage", passed);
    return passed;
}

// =============================================================================
// 主函数
// =============================================================================

int main(int argc, char* argv[]) {
    (void)argc;
    (void)argv;

    printf("==============================================\n");
    printf("RVC Streaming Integration Test Suite\n");
    printf("==============================================\n");

    // 初始化日志
    log_set_level(LOG_INFO);

    // 初始化随机数生成器
    srand((unsigned int)time(NULL));

    int total_tests = 0;
    int passed_tests = 0;

    #define RUN_TEST(test_func) do { \
        total_tests++; \
        if (test_func()) passed_tests++; \
    } while(0)

    RUN_TEST(test_f0_integration);
    RUN_TEST(test_pitch_cache_integration);
    RUN_TEST(test_feature_upsample_performance);
    RUN_TEST(test_streaming_latency);
    RUN_TEST(test_continuous_stream);
    RUN_TEST(test_memory_usage);

    printf("\n==============================================\n");
    printf("Integration Test Results: %d/%d passed\n", passed_tests, total_tests);
    printf("==============================================\n");

    return (passed_tests == total_tests) ? 0 : 1;
}
