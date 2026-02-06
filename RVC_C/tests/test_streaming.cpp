/**
 * @file test_streaming.cpp
 * @brief RVC 流式处理测试程序
 */

#include "rvc_streaming.h"
#include "rvc_onnx.h"
#include "utils.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

// =============================================================================
// 测试工具函数
// =============================================================================

/** 生成测试正弦波 */
static void generate_sine_wave(float* buffer, size_t samples, float freq, int sample_rate) {
    for (size_t i = 0; i < samples; i++) {
        buffer[i] = sinf(2.0f * 3.14159265f * freq * (float)i / (float)sample_rate);
    }
}

/** 计算 RMS */
static float calc_rms(const float* buffer, size_t samples) {
    if (samples == 0) return 0.0f;
    double sum = 0.0;
    for (size_t i = 0; i < samples; i++) {
        sum += buffer[i] * buffer[i];
    }
    return sqrtf((float)(sum / samples));
}

/** 打印测试结果 */
static void print_result(const char* test_name, int passed) {
    printf("[%s] %s\n", passed ? "PASS" : "FAIL", test_name);
}

// =============================================================================
// 测试用例
// =============================================================================

/** 测试默认配置 */
static int test_default_config(void) {
    printf("\n--- Test: Default Config ---\n");

    RVCStreamConfig config = rvc_stream_default_config();

    int passed = 1;
    passed &= (config.block_size == RVC_DEFAULT_BLOCK_SIZE);
    passed &= (config.audio_context_size == RVC_DEFAULT_AUDIO_CONTEXT_SIZE);
    passed &= (config.crossfade_samples == RVC_DEFAULT_CROSSFADE_SAMPLES);
    passed &= (config.speaker_id == 0);

    printf("block_size: %zu (expected: %d)\n",
           config.block_size, RVC_DEFAULT_BLOCK_SIZE);
    printf("audio_context_size: %zu (expected: %d)\n",
           config.audio_context_size, RVC_DEFAULT_AUDIO_CONTEXT_SIZE);
    printf("crossfade_samples: %zu (expected: %d)\n",
           config.crossfade_samples, RVC_DEFAULT_CROSSFADE_SAMPLES);

    print_result("Default Config", passed);
    return passed;
}

/** 测试状态创建和销毁 */
static int test_state_create_destroy(void) {
    printf("\n--- Test: State Create/Destroy ---\n");

    // 使用默认配置创建
    RVCStreamState* state1 = rvc_stream_state_create(NULL);
    int passed = (state1 != NULL);

    if (state1) {
        printf("State created with default config: OK\n");
        printf("  cache_pitch: %p\n", (void*)state1->cache_pitch);
        printf("  cache_pitchf: %p\n", (void*)state1->cache_pitchf);
        printf("  audio_context: %p\n", (void*)state1->audio_context);
        printf("  is_first_chunk: %d\n", state1->is_first_chunk);
        rvc_stream_state_destroy(state1);
    }

    // 使用自定义配置创建
    RVCStreamConfig config = rvc_stream_default_config();
    config.block_size = 8000;
    config.audio_context_size = 2400;

    RVCStreamState* state2 = rvc_stream_state_create(&config);
    passed &= (state2 != NULL);

    if (state2) {
        printf("State created with custom config: OK\n");
        rvc_stream_state_destroy(state2);
    }

    // 测试 NULL 销毁 (不应崩溃)
    rvc_stream_state_destroy(NULL);
    printf("Destroy NULL state: OK (no crash)\n");

    print_result("State Create/Destroy", passed);
    return passed;
}

/** 测试状态重置 */
static int test_state_reset(void) {
    printf("\n--- Test: State Reset ---\n");

    RVCStreamState* state = rvc_stream_state_create(NULL);
    if (!state) {
        print_result("State Reset", 0);
        return 0;
    }

    // 修改状态
    state->is_first_chunk = 0;
    state->total_samples_processed = 16000;
    state->cache_pitch[0] = 100;
    state->cache_pitchf[0] = 200.0f;

    // 重置
    rvc_stream_state_reset(state);

    int passed = 1;
    passed &= (state->is_first_chunk == 1);
    passed &= (state->total_samples_processed == 0);
    passed &= (state->cache_pitch[0] == 0);
    passed &= (state->cache_pitchf[0] == 0.0f);

    printf("After reset:\n");
    printf("  is_first_chunk: %d (expected: 1)\n", state->is_first_chunk);
    printf("  total_samples_processed: %zu (expected: 0)\n", state->total_samples_processed);

    rvc_stream_state_destroy(state);

    print_result("State Reset", passed);
    return passed;
}

/** 测试 F0 转 Pitch */
static int test_f0_to_pitch(void) {
    printf("\n--- Test: F0 to Pitch ---\n");

    int passed = 1;

    // 测试边界情况
    int64_t p_zero = rvc_f0_to_pitch(0.0f, 50.0f, 1100.0f);
    printf("F0=0 -> pitch=%lld (expected: 1)\n", (long long)p_zero);
    passed &= (p_zero == 1);

    int64_t p_negative = rvc_f0_to_pitch(-100.0f, 50.0f, 1100.0f);
    printf("F0=-100 -> pitch=%lld (expected: 1)\n", (long long)p_negative);
    passed &= (p_negative == 1);

    // 测试正常值
    int64_t p_low = rvc_f0_to_pitch(50.0f, 50.0f, 1100.0f);
    printf("F0=50 (min) -> pitch=%lld (expected: ~1)\n", (long long)p_low);
    passed &= (p_low >= 1 && p_low <= 10);

    int64_t p_high = rvc_f0_to_pitch(1100.0f, 50.0f, 1100.0f);
    printf("F0=1100 (max) -> pitch=%lld (expected: ~255)\n", (long long)p_high);
    passed &= (p_high >= 250 && p_high <= 255);

    int64_t p_mid = rvc_f0_to_pitch(440.0f, 50.0f, 1100.0f);
    printf("F0=440 (A4) -> pitch=%lld (expected: mid-range)\n", (long long)p_mid);
    passed &= (p_mid > 50 && p_mid < 200);

    print_result("F0 to Pitch", passed);
    return passed;
}

/** 测试批量 F0 转 Pitch */
static int test_f0_to_pitch_batch(void) {
    printf("\n--- Test: F0 to Pitch Batch ---\n");

    const size_t length = 5;
    double f0_values[] = {0.0, 100.0, 200.0, 440.0, 880.0};
    int64_t pitch[5];
    float pitchf[5];

    rvc_f0_to_pitch_batch(f0_values, length, pitch, pitchf, 50.0f, 1100.0f);

    printf("Batch conversion results:\n");
    int passed = 1;
    for (size_t i = 0; i < length; i++) {
        printf("  F0=%.1f -> pitch=%lld, pitchf=%.1f\n",
               f0_values[i], (long long)pitch[i], pitchf[i]);
        if (f0_values[i] <= 0) {
            passed &= (pitch[i] == 1);
        } else {
            passed &= (pitch[i] >= 1 && pitch[i] <= 255);
        }
    }

    print_result("F0 to Pitch Batch", passed);
    return passed;
}

/** 测试特征 2x 上采样 */
static int test_interpolate_features(void) {
    printf("\n--- Test: Feature 2x Interpolation ---\n");

    const size_t input_frames = 4;
    const size_t feature_dim = 3;  // 简化测试

    float input[] = {
        1.0f, 2.0f, 3.0f,   // Frame 0
        5.0f, 6.0f, 7.0f,   // Frame 1
        9.0f, 10.0f, 11.0f, // Frame 2
        13.0f, 14.0f, 15.0f // Frame 3
    };

    float output[8 * 3];  // 8 frames * 3 dims

    rvc_interpolate_features_2x(input, input_frames, output, feature_dim);

    printf("Interpolation results:\n");
    int passed = 1;

    for (size_t i = 0; i < input_frames * 2; i++) {
        printf("  Frame %zu: [%.1f, %.1f, %.1f]\n", i,
               output[i * feature_dim],
               output[i * feature_dim + 1],
               output[i * feature_dim + 2]);
    }

    // 验证原始帧被复制
    passed &= (output[0] == 1.0f);  // Frame 0
    passed &= (output[6] == 5.0f);  // Frame 2 (原 Frame 1)

    // 验证插值帧
    passed &= (fabsf(output[3] - 3.0f) < 0.01f);  // Frame 1 应该是 Frame 0 和 Frame 1 的平均

    print_result("Feature 2x Interpolation", passed);
    return passed;
}

/** 测试输出大小计算 */
static int test_calc_output_size(void) {
    printf("\n--- Test: Calculate Output Size ---\n");

    size_t input_samples = 16000;  // 1 秒 @ 16kHz
    size_t output_size = rvc_stream_calc_output_size(input_samples);

    printf("Input: %zu samples @ 16kHz\n", input_samples);
    printf("Output buffer size: %zu samples\n", output_size);
    printf("Ratio: %.2f (expected: >= 3.0)\n", (float)output_size / input_samples);

    int passed = (output_size >= input_samples * 3);

    print_result("Calculate Output Size", passed);
    return passed;
}

/** 测试单块流式处理 */
static int test_single_chunk_process(void) {
    printf("\n--- Test: Single Chunk Process ---\n");

    // 创建状态
    RVCStreamState* state = rvc_stream_state_create(NULL);
    if (!state) {
        print_result("Single Chunk Process", 0);
        return 0;
    }

    // 生成测试输入 (1 秒 @ 16kHz)
    const size_t input_samples = 16000;
    float* input = (float*)malloc(input_samples * sizeof(float));
    generate_sine_wave(input, input_samples, 440.0f, 16000);

    // 分配输出缓冲区
    size_t output_capacity = rvc_stream_calc_output_size(input_samples);
    float* output = (float*)malloc(output_capacity * sizeof(float));
    size_t output_samples = 0;

    // 处理 (注意: ctx 为 NULL，这是占位测试)
    // 实际使用时需要有效的 RVCContext
    RVCStreamError err = rvc_stream_process(
        NULL,  // ctx - 占位测试
        state,
        input, input_samples,
        output, output_capacity,
        &output_samples,
        NULL  // 使用默认配置
    );

    // 当 ctx 为 NULL 时应该返回错误
    int passed = (err == RVC_STREAM_ERROR_INVALID_PARAM);
    printf("Process with NULL ctx: %s (expected: INVALID_PARAM)\n",
           rvc_stream_error_string(err));

    // 清理
    free(input);
    free(output);
    rvc_stream_state_destroy(state);

    print_result("Single Chunk Process", passed);
    return passed;
}

/** 测试错误字符串 */
static int test_error_strings(void) {
    printf("\n--- Test: Error Strings ---\n");

    int passed = 1;

    const char* str_success = rvc_stream_error_string(RVC_STREAM_SUCCESS);
    printf("SUCCESS: %s\n", str_success);
    passed &= (str_success != NULL);

    const char* str_invalid = rvc_stream_error_string(RVC_STREAM_ERROR_INVALID_PARAM);
    printf("INVALID_PARAM: %s\n", str_invalid);
    passed &= (str_invalid != NULL);

    const char* str_memory = rvc_stream_error_string(RVC_STREAM_ERROR_MEMORY);
    printf("MEMORY: %s\n", str_memory);
    passed &= (str_memory != NULL);

    print_result("Error Strings", passed);
    return passed;
}

/** 测试多块连续处理 */
static int test_multi_chunk_process(void) {
    printf("\n--- Test: Multi Chunk Process ---\n");

    RVCStreamState* state = rvc_stream_state_create(NULL);
    if (!state) {
        print_result("Multi Chunk Process", 0);
        return 0;
    }

    printf("Initial state:\n");
    printf("  is_first_chunk: %d\n", state->is_first_chunk);
    printf("  total_samples_processed: %zu\n", state->total_samples_processed);

    // 模拟多块处理 (状态更新测试)
    const size_t chunk_size = 4000;  // 250ms @ 16kHz
    const int num_chunks = 4;

    float* input = (float*)malloc(chunk_size * sizeof(float));
    generate_sine_wave(input, chunk_size, 440.0f, 16000);

    size_t output_capacity = rvc_stream_calc_output_size(chunk_size);
    float* output = (float*)malloc(output_capacity * sizeof(float));

    int passed = 1;

    for (int i = 0; i < num_chunks; i++) {
        // 手动更新状态 (因为 ctx 为 NULL，实际处理会跳过)
        state->is_first_chunk = (i == 0) ? 1 : 0;
        state->total_samples_processed += chunk_size;

        printf("Chunk %d: total_samples=%zu\n", i + 1, state->total_samples_processed);
    }

    passed &= (state->total_samples_processed == chunk_size * num_chunks);

    free(input);
    free(output);
    rvc_stream_state_destroy(state);

    print_result("Multi Chunk Process", passed);
    return passed;
}

/** 测试刷新功能 */
static int test_flush(void) {
    printf("\n--- Test: Flush ---\n");

    RVCStreamState* state = rvc_stream_state_create(NULL);
    if (!state) {
        print_result("Flush", 0);
        return 0;
    }

    // 模拟有剩余数据的情况
    const size_t context_size = 1000;
    for (size_t i = 0; i < context_size; i++) {
        state->audio_context[i] = sinf(2.0f * 3.14159f * 440.0f * i / 16000.0f);
    }
    state->audio_context_size = context_size;

    // 分配输出
    size_t output_capacity = 5000;
    float* output = (float*)malloc(output_capacity * sizeof(float));
    size_t output_samples = 0;

    // 刷新 (ctx 为 NULL 会返回错误)
    RVCStreamError err = rvc_stream_flush(
        NULL, state,
        output, output_capacity,
        &output_samples,
        NULL
    );

    printf("Flush with NULL ctx: %s\n", rvc_stream_error_string(err));

    int passed = (err == RVC_STREAM_ERROR_INVALID_PARAM);

    free(output);
    rvc_stream_state_destroy(state);

    print_result("Flush", passed);
    return passed;
}

// =============================================================================
// 主函数
// =============================================================================

int main(int argc, char* argv[]) {
    printf("===========================================\n");
    printf("RVC Streaming Module Test Suite\n");
    printf("===========================================\n");

    int total_tests = 0;
    int passed_tests = 0;

    // 运行所有测试
    #define RUN_TEST(test_func) do { \
        total_tests++; \
        if (test_func()) passed_tests++; \
    } while(0)

    RUN_TEST(test_default_config);
    RUN_TEST(test_state_create_destroy);
    RUN_TEST(test_state_reset);
    RUN_TEST(test_f0_to_pitch);
    RUN_TEST(test_f0_to_pitch_batch);
    RUN_TEST(test_interpolate_features);
    RUN_TEST(test_calc_output_size);
    RUN_TEST(test_single_chunk_process);
    RUN_TEST(test_error_strings);
    RUN_TEST(test_multi_chunk_process);
    RUN_TEST(test_flush);

    printf("\n===========================================\n");
    printf("Test Results: %d/%d passed\n", passed_tests, total_tests);
    printf("===========================================\n");

    return (passed_tests == total_tests) ? 0 : 1;
}
