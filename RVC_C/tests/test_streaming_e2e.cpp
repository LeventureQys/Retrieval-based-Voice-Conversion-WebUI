/**
 * @file test_streaming_e2e.cpp
 * @brief RVC 流式处理端到端测试
 *
 * 使用真实 ONNX 模型测试完整的流式语音转换流程。
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "rvc_streaming.h"
#include "onnx_inference.h"
#include "audio_processor.h"
#include "f0_extractor.h"
#include "utils.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/**
 * 流式语音转换测试
 */
int test_streaming_conversion(
    const char* contentvec_path,
    const char* synthesizer_path,
    const char* input_wav,
    const char* output_wav,
    int speaker_id,
    float pitch_shift,
    size_t block_size_ms
) {
    printf("\n========================================\n");
    printf("   RVC Streaming Voice Conversion Test\n");
    printf("========================================\n");

    int ret = 0;

    // 计算块大小 (采样点数)
    size_t block_size = (block_size_ms * 16000) / 1000;
    printf("Block size: %zu ms (%zu samples @ 16kHz)\n", block_size_ms, block_size);

    // =================================================================
    // 步骤 1: 创建 ONNX 引擎和会话
    // =================================================================
    printf("\n[1/7] Creating ONNX engine...\n");
    ONNXEngine* engine = onnx_engine_create(4);
    if (!engine) {
        printf("[FAIL] Failed to create ONNX engine\n");
        return -1;
    }

    printf("[2/7] Loading ContentVec model...\n");
    ONNXSession* contentvec_session = onnx_session_create(engine, contentvec_path);
    if (!contentvec_session) {
        printf("[FAIL] Failed to load ContentVec: %s\n", contentvec_path);
        onnx_engine_destroy(engine);
        return -1;
    }
    printf("[PASS] ContentVec loaded\n");

    printf("[3/7] Loading Synthesizer model...\n");
    ONNXSession* synth_session = onnx_session_create(engine, synthesizer_path);
    if (!synth_session) {
        printf("[FAIL] Failed to load Synthesizer: %s\n", synthesizer_path);
        onnx_session_destroy(contentvec_session);
        onnx_engine_destroy(engine);
        return -1;
    }
    printf("[PASS] Synthesizer loaded\n");

    // =================================================================
    // 步骤 2: 创建 F0 提取器
    // =================================================================
    printf("[4/7] Creating F0 extractor...\n");
    F0Extractor* f0_extractor = f0_extractor_create(F0_METHOD_HARVEST, 16000);
    if (!f0_extractor) {
        printf("[FAIL] Failed to create F0 extractor\n");
        onnx_session_destroy(synth_session);
        onnx_session_destroy(contentvec_session);
        onnx_engine_destroy(engine);
        return -1;
    }

    // 设置 F0 参数
    F0Params f0_params = f0_default_params();
    f0_params.frame_period = 10.0;  // 10ms
    f0_params.f0_floor = 50.0;
    f0_params.f0_ceil = 1100.0;
    f0_extractor_set_params(f0_extractor, &f0_params);
    printf("[PASS] F0 extractor created\n");

    // =================================================================
    // 步骤 3: 加载输入音频
    // =================================================================
    printf("[5/7] Loading input audio...\n");
    AudioBuffer input_buffer = audio_buffer_create(0);
    AudioFormat input_format;

    ret = audio_load_file(input_wav, &input_buffer, &input_format);
    if (ret != 0) {
        printf("[FAIL] Failed to load: %s\n", input_wav);
        f0_extractor_destroy(f0_extractor);
        onnx_session_destroy(synth_session);
        onnx_session_destroy(contentvec_session);
        onnx_engine_destroy(engine);
        return -1;
    }
    printf("[PASS] Loaded: %zu samples, %d Hz\n", input_buffer.size, input_format.sample_rate);

    // 重采样到 16kHz
    float* audio_16k = NULL;
    size_t audio_16k_size = 0;

    if (input_format.sample_rate != 16000) {
        printf("       Resampling to 16kHz...\n");
        ret = audio_resample(input_buffer.data, input_buffer.size,
                            input_format.sample_rate, 16000,
                            &audio_16k, &audio_16k_size);
        if (ret != 0) {
            printf("[FAIL] Resampling failed\n");
            audio_buffer_free(&input_buffer);
            f0_extractor_destroy(f0_extractor);
            onnx_session_destroy(synth_session);
            onnx_session_destroy(contentvec_session);
            onnx_engine_destroy(engine);
            return -1;
        }
        printf("       Resampled: %zu samples\n", audio_16k_size);
    } else {
        audio_16k = input_buffer.data;
        audio_16k_size = input_buffer.size;
    }

    // =================================================================
    // 步骤 4: 创建流式状态
    // =================================================================
    printf("[6/7] Creating streaming state...\n");
    RVCStreamConfig stream_config = rvc_stream_default_config();
    stream_config.block_size = block_size;
    stream_config.speaker_id = speaker_id;

    RVCStreamState* state = rvc_stream_state_create(&stream_config);
    if (!state) {
        printf("[FAIL] Failed to create streaming state\n");
        if (audio_16k != input_buffer.data) free(audio_16k);
        audio_buffer_free(&input_buffer);
        f0_extractor_destroy(f0_extractor);
        onnx_session_destroy(synth_session);
        onnx_session_destroy(contentvec_session);
        onnx_engine_destroy(engine);
        return -1;
    }
    printf("[PASS] Streaming state created\n");

    // =================================================================
    // 步骤 5: 流式处理
    // =================================================================
    printf("[7/7] Processing audio in streaming mode...\n");

    // 计算块数
    size_t num_chunks = (audio_16k_size + block_size - 1) / block_size;
    printf("       Total chunks: %zu\n", num_chunks);

    // 分配输出缓冲区
    size_t total_output_capacity = audio_16k_size * 4;  // 预留足够空间
    float* total_output = (float*)calloc(total_output_capacity, sizeof(float));
    size_t total_output_size = 0;

    size_t chunk_output_capacity = rvc_stream_calc_output_size(block_size);
    float* chunk_output = (float*)malloc(chunk_output_capacity * sizeof(float));

    if (!total_output || !chunk_output) {
        printf("[FAIL] Memory allocation failed\n");
        ret = -1;
        goto cleanup;
    }

    int64_t total_start = get_time_ms();

    for (size_t chunk_idx = 0; chunk_idx < num_chunks; chunk_idx++) {
        size_t chunk_start = chunk_idx * block_size;
        size_t chunk_samples = block_size;
        if (chunk_start + chunk_samples > audio_16k_size) {
            chunk_samples = audio_16k_size - chunk_start;
        }

        const float* chunk_input = audio_16k + chunk_start;
        size_t chunk_output_size = 0;

        int64_t chunk_start_time = get_time_ms();

        RVCStreamError err = rvc_stream_process_standalone(
            contentvec_session,
            synth_session,
            f0_extractor,
            state,
            chunk_input,
            chunk_samples,
            chunk_output,
            chunk_output_capacity,
            &chunk_output_size,
            speaker_id,
            pitch_shift
        );

        int64_t chunk_elapsed = get_time_ms() - chunk_start_time;

        if (err != RVC_STREAM_SUCCESS) {
            printf("[FAIL] Chunk %zu failed: %s\n", chunk_idx, rvc_stream_error_string(err));
            ret = -1;
            goto cleanup;
        }

        // 复制到总输出
        if (total_output_size + chunk_output_size <= total_output_capacity) {
            memcpy(total_output + total_output_size, chunk_output, chunk_output_size * sizeof(float));
            total_output_size += chunk_output_size;
        }

        // 计算 RTF
        double audio_duration_ms = (double)chunk_samples / 16.0;  // ms
        double rtf = (double)chunk_elapsed / audio_duration_ms;

        printf("       Chunk %zu/%zu: %zu -> %zu samples, %lld ms (RTF: %.2f)\n",
               chunk_idx + 1, num_chunks,
               chunk_samples, chunk_output_size,
               (long long)chunk_elapsed, rtf);
    }

    int64_t total_elapsed = get_time_ms() - total_start;
    double total_audio_duration = (double)audio_16k_size / 16000.0;
    double total_rtf = ((double)total_elapsed / 1000.0) / total_audio_duration;

    printf("\n[PASS] Streaming completed!\n");
    printf("       Total time: %lld ms\n", (long long)total_elapsed);
    printf("       Audio duration: %.2f s\n", total_audio_duration);
    printf("       Overall RTF: %.2f\n", total_rtf);
    printf("       Output samples: %zu\n", total_output_size);

    // =================================================================
    // 步骤 6: 保存输出
    // =================================================================
    {
        AudioBuffer out_buffer;
        out_buffer.data = total_output;
        out_buffer.size = total_output_size;
        out_buffer.capacity = total_output_size;

        AudioFormat out_format;
        out_format.sample_rate = 48000;
        out_format.channels = 1;
        out_format.bits_per_sample = 32;

        ret = audio_save_file(output_wav, &out_buffer, &out_format);
        if (ret == 0) {
            printf("[PASS] Saved output to: %s\n", output_wav);
        } else {
            printf("[FAIL] Failed to save output\n");
        }
    }

cleanup:
    free(chunk_output);
    free(total_output);
    rvc_stream_state_destroy(state);
    if (audio_16k != input_buffer.data) free(audio_16k);
    audio_buffer_free(&input_buffer);
    f0_extractor_destroy(f0_extractor);
    onnx_session_destroy(synth_session);
    onnx_session_destroy(contentvec_session);
    onnx_engine_destroy(engine);

    printf("\n========================================\n");
    printf("   %s\n", ret == 0 ? "Test PASSED!" : "Test FAILED!");
    printf("========================================\n");

    return ret;
}

void print_usage(const char* prog) {
    printf("RVC Streaming Voice Conversion Test\n\n");
    printf("Usage: %s -c <contentvec.onnx> -s <synthesizer.onnx> -i <input.wav> [options]\n\n", prog);
    printf("Required:\n");
    printf("  -c <path>   Path to ContentVec/HuBERT ONNX model\n");
    printf("  -s <path>   Path to Synthesizer ONNX model\n");
    printf("  -i <path>   Path to input WAV file\n");
    printf("\nOptions:\n");
    printf("  -o <path>   Path to output WAV file (default: streaming_output.wav)\n");
    printf("  -p <float>  Pitch shift in semitones (default: 0)\n");
    printf("  -sid <int>  Speaker ID (default: 0)\n");
    printf("  -b <int>    Block size in milliseconds (default: 500)\n");
}

int main(int argc, char* argv[]) {
    const char* contentvec_path = NULL;
    const char* synthesizer_path = NULL;
    const char* input_wav = NULL;
    const char* output_wav = "streaming_output.wav";
    int speaker_id = 0;
    float pitch_shift = 0.0f;
    size_t block_size_ms = 500;  // 500ms 默认

    // 解析命令行参数
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-c") == 0 && i + 1 < argc) {
            contentvec_path = argv[++i];
        } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            synthesizer_path = argv[++i];
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            input_wav = argv[++i];
        } else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            output_wav = argv[++i];
        } else if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) {
            pitch_shift = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "-sid") == 0 && i + 1 < argc) {
            speaker_id = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-b") == 0 && i + 1 < argc) {
            block_size_ms = (size_t)atoi(argv[++i]);
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }

    if (!contentvec_path || !synthesizer_path || !input_wav) {
        print_usage(argv[0]);
        return 1;
    }

    log_set_level(LOG_INFO);

    return test_streaming_conversion(
        contentvec_path,
        synthesizer_path,
        input_wav,
        output_wav,
        speaker_id,
        pitch_shift,
        block_size_ms
    );
}
