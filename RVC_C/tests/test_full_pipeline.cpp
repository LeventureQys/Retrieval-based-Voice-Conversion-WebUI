/**
 * @file test_full_pipeline.cpp
 * @brief 完整的 RVC 语音转换流程测试
 *
 * 使用 ContentVec + Synthesizer 实现端到端语音转换
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <algorithm>

#include "onnx_inference.h"
#include "audio_processor.h"
#include "f0_extractor.h"
#include "utils.h"
#include "mt19937.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/**
 * 完整的 RVC 语音转换
 */
int rvc_voice_convert(
    const char* contentvec_path,
    const char* synthesizer_path,
    const char* input_wav,
    const char* output_wav,
    int speaker_id,
    float pitch_shift
) {
    printf("\n========================================\n");
    printf("   RVC Voice Conversion\n");
    printf("========================================\n");

    int ret = 0;

    // 创建 ONNX 引擎
    ONNXEngine* engine = onnx_engine_create(4);
    if (!engine) {
        printf("[FAIL] Failed to create ONNX engine\n");
        return -1;
    }

    // 加载 ContentVec 模型
    printf("\n[1/6] Loading ContentVec model...\n");
    ONNXSession* contentvec_session = onnx_session_create(engine, contentvec_path);
    if (!contentvec_session) {
        printf("[FAIL] Failed to load ContentVec model\n");
        onnx_engine_destroy(engine);
        return -1;
    }
    printf("[PASS] ContentVec model loaded\n");

    // 加载合成器模型
    printf("\n[2/6] Loading Synthesizer model...\n");
    ONNXSession* synth_session = onnx_session_create(engine, synthesizer_path);
    if (!synth_session) {
        printf("[FAIL] Failed to load Synthesizer model\n");
        onnx_session_destroy(contentvec_session);
        onnx_engine_destroy(engine);
        return -1;
    }
    printf("[PASS] Synthesizer model loaded\n");

    // 加载输入音频
    printf("\n[3/6] Loading input audio...\n");
    AudioBuffer input_buffer = audio_buffer_create(0);
    AudioFormat input_format;

    ret = audio_load_file(input_wav, &input_buffer, &input_format);
    if (ret != 0) {
        printf("[FAIL] Failed to load input audio: %s\n", input_wav);
        onnx_session_destroy(synth_session);
        onnx_session_destroy(contentvec_session);
        onnx_engine_destroy(engine);
        return -1;
    }
    printf("[PASS] Loaded: %zu samples, %d Hz, %d channels\n",
           input_buffer.size, input_format.sample_rate, input_format.channels);

    // 重采样到 16kHz (ContentVec 需要)
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

    // 提取 F0
    printf("\n[4/6] Extracting F0...\n");
    F0Extractor* f0_extractor = f0_extractor_create(F0_METHOD_HARVEST, input_format.sample_rate);
    if (!f0_extractor) {
        printf("[FAIL] Failed to create F0 extractor\n");
        if (audio_16k != input_buffer.data) free(audio_16k);
        audio_buffer_free(&input_buffer);
        onnx_session_destroy(synth_session);
        onnx_session_destroy(contentvec_session);
        onnx_engine_destroy(engine);
        return -1;
    }

    // 设置 F0 提取参数
    // frame_period = 1000 * hop_size / sample_rate
    // 对于 RVC: hop_size = 512, sample_rate = 48000
    // frame_period = 1000 * 512 / 48000 = 10.67 ms
    F0Params f0_params = f0_default_params();
    f0_params.frame_period = 1000.0 * 512 / input_format.sample_rate;
    f0_params.f0_floor = 50.0;
    f0_params.f0_ceil = 1100.0;
    f0_extractor_set_params(f0_extractor, &f0_params);
    printf("       F0 frame_period: %.2f ms\n", f0_params.frame_period);

    F0Result f0_result;
    memset(&f0_result, 0, sizeof(F0Result));

    int64_t f0_start = get_time_ms();
    ret = f0_extract_float(f0_extractor, input_buffer.data, input_buffer.size, &f0_result);
    int64_t f0_elapsed = get_time_ms() - f0_start;

    if (ret != 0) {
        printf("[FAIL] F0 extraction failed\n");
        f0_extractor_destroy(f0_extractor);
        if (audio_16k != input_buffer.data) free(audio_16k);
        audio_buffer_free(&input_buffer);
        onnx_session_destroy(synth_session);
        onnx_session_destroy(contentvec_session);
        onnx_engine_destroy(engine);
        return -1;
    }
    printf("[PASS] F0 extracted: %zu frames in %lld ms\n", f0_result.length, (long long)f0_elapsed);

    // 注意: 音高偏移将在 F0 resize 和 interpolate 之后应用

    // 提取 ContentVec 特征
    printf("\n[5/6] Extracting ContentVec features...\n");

    // 准备输入: [1, 1, audio_length]
    int64_t cv_input_dims[] = {1, 1, (int64_t)audio_16k_size};
    TensorData cv_input;
    cv_input.data = audio_16k;
    cv_input.size = audio_16k_size;
    cv_input.shape = tensor_shape_create(cv_input_dims, 3);
    cv_input.dtype = TENSOR_TYPE_FLOAT32;

    TensorData* cv_outputs = NULL;
    size_t cv_num_outputs = 0;

    int64_t cv_start = get_time_ms();
    ret = onnx_session_run_multi(contentvec_session, &cv_input, 1, &cv_outputs, &cv_num_outputs);
    int64_t cv_elapsed = get_time_ms() - cv_start;

    tensor_shape_free(&cv_input.shape);

    if (ret != 0 || cv_num_outputs == 0) {
        printf("[FAIL] ContentVec inference failed\n");
        f0_result_free(&f0_result);
        f0_extractor_destroy(f0_extractor);
        if (audio_16k != input_buffer.data) free(audio_16k);
        audio_buffer_free(&input_buffer);
        onnx_session_destroy(synth_session);
        onnx_session_destroy(contentvec_session);
        onnx_engine_destroy(engine);
        return -1;
    }

    // ContentVec 输出: [1, time_frames, 768]
    size_t cv_frames = cv_outputs[0].shape.dims[1];
    size_t cv_dim = cv_outputs[0].shape.dims[2];
    printf("[PASS] ContentVec features: %zu frames x %zu dims in %lld ms\n",
           cv_frames, cv_dim, (long long)cv_elapsed);

    // 准备合成器输入
    printf("\n[6/6] Running synthesizer...\n");

    // 需要将 ContentVec 特征重复 2 倍以匹配合成器帧率
    // ContentVec hop_size = 320, 合成器需要 2x 上采样
    size_t synth_frames = cv_frames * 2;

    printf("       Synth frames: %zu (ContentVec frames * 2)\n", synth_frames);
    printf("       Original F0 frames: %zu\n", f0_result.length);

    // 将 F0 resize 到 synth_frames 长度
    double* f0_resized = (double*)malloc(synth_frames * sizeof(double));
    if (!f0_resized) {
        printf("[FAIL] Memory allocation failed\n");
        ret = -1;
        goto cleanup;
    }

    f0_resize(f0_result.f0, f0_result.length, synth_frames, f0_resized);
    printf("       F0 resized to: %zu frames\n", synth_frames);

    // 对 F0 进行插值处理，填充无声段
    f0_interpolate(f0_resized, synth_frames);
    printf("       F0 interpolated\n");

    // 应用音高偏移 (在 resize 和 interpolate 之后)
    if (pitch_shift != 0.0f) {
        f0_shift_pitch(f0_resized, synth_frames, pitch_shift);
        printf("       Pitch shift applied: %.1f semitones\n", pitch_shift);
    }

    // 分配合成器输入
    float* phone = (float*)malloc(synth_frames * cv_dim * sizeof(float));
    int64_t* pitch = (int64_t*)malloc(synth_frames * sizeof(int64_t));
    float* pitchf = (float*)malloc(synth_frames * sizeof(float));
    float* rnd = (float*)malloc(192 * synth_frames * sizeof(float));

    if (!phone || !pitch || !pitchf || !rnd) {
        printf("[FAIL] Memory allocation failed\n");
        ret = -1;
        free(f0_resized);
        goto cleanup;
    }

    // 复制并重复 ContentVec 特征
    float* cv_data = (float*)cv_outputs[0].data;
    for (size_t t = 0; t < synth_frames; t++) {
        size_t src_t = t / 2;
        if (src_t >= cv_frames) src_t = cv_frames - 1;
        memcpy(&phone[t * cv_dim], &cv_data[src_t * cv_dim], cv_dim * sizeof(float));
    }

    // 准备音高数据 (使用 resized 和 interpolated 的 F0)
    float f0_min = 50.0f;
    float f0_max = 1100.0f;
    float f0_mel_min = 1127.0f * logf(1.0f + f0_min / 700.0f);
    float f0_mel_max = 1127.0f * logf(1.0f + f0_max / 700.0f);

    for (size_t t = 0; t < synth_frames; t++) {
        float f0_val = (float)f0_resized[t];
        pitchf[t] = f0_val;

        if (f0_val > 0) {
            float f0_mel = 1127.0f * logf(1.0f + f0_val / 700.0f);
            f0_mel = (f0_mel - f0_mel_min) * 254.0f / (f0_mel_max - f0_mel_min) + 1.0f;
            if (f0_mel < 1.0f) f0_mel = 1.0f;
            if (f0_mel > 255.0f) f0_mel = 255.0f;
            pitch[t] = (int64_t)roundf(f0_mel);
        } else {
            pitch[t] = 1;  // 使用 1 而不是 0，与 Python 一致
        }
    }

    // 释放 f0_resized
    free(f0_resized);
    f0_resized = NULL;

    // 生成随机噪声 (使用 Mersenne Twister，与 numpy.random.randn 兼容)
    MT19937State mt_state;
    mt19937_seed(&mt_state, 42);
    for (size_t i = 0; i < 192 * synth_frames; i++) {
        rnd[i] = mt19937_randn_float(&mt_state);
    }

    // 打印一些调试信息
    printf("       Phone range: [%.4f, %.4f]\n",
           *std::min_element(phone, phone + synth_frames * cv_dim),
           *std::max_element(phone, phone + synth_frames * cv_dim));

    int64_t pitch_min = *std::min_element(pitch, pitch + synth_frames);
    int64_t pitch_max = *std::max_element(pitch, pitch + synth_frames);
    printf("       Pitch range: [%lld, %lld]\n", (long long)pitch_min, (long long)pitch_max);

    float pitchf_min = *std::min_element(pitchf, pitchf + synth_frames);
    float pitchf_max = *std::max_element(pitchf, pitchf + synth_frames);
    printf("       Pitchf range: [%.1f, %.1f]\n", pitchf_min, pitchf_max);

    // 准备合成器输入张量
    TensorData synth_inputs[6];
    int64_t phone_lengths_val = (int64_t)synth_frames;
    int64_t ds_val = (int64_t)speaker_id;

    // phone: [1, synth_frames, 768]
    int64_t phone_dims[] = {1, (int64_t)synth_frames, (int64_t)cv_dim};
    synth_inputs[0].data = phone;
    synth_inputs[0].size = synth_frames * cv_dim;
    synth_inputs[0].shape = tensor_shape_create(phone_dims, 3);
    synth_inputs[0].dtype = TENSOR_TYPE_FLOAT32;

    // phone_lengths: [1]
    int64_t phone_lengths_dims[] = {1};
    synth_inputs[1].data = &phone_lengths_val;
    synth_inputs[1].size = 1;
    synth_inputs[1].shape = tensor_shape_create(phone_lengths_dims, 1);
    synth_inputs[1].dtype = TENSOR_TYPE_INT64;

    // pitch: [1, synth_frames]
    int64_t pitch_dims[] = {1, (int64_t)synth_frames};
    synth_inputs[2].data = pitch;
    synth_inputs[2].size = synth_frames;
    synth_inputs[2].shape = tensor_shape_create(pitch_dims, 2);
    synth_inputs[2].dtype = TENSOR_TYPE_INT64;

    // pitchf: [1, synth_frames]
    int64_t pitchf_dims[] = {1, (int64_t)synth_frames};
    synth_inputs[3].data = pitchf;
    synth_inputs[3].size = synth_frames;
    synth_inputs[3].shape = tensor_shape_create(pitchf_dims, 2);
    synth_inputs[3].dtype = TENSOR_TYPE_FLOAT32;

    // ds: [1]
    int64_t ds_dims[] = {1};
    synth_inputs[4].data = &ds_val;
    synth_inputs[4].size = 1;
    synth_inputs[4].shape = tensor_shape_create(ds_dims, 1);
    synth_inputs[4].dtype = TENSOR_TYPE_INT64;

    // rnd: [1, 192, synth_frames]
    int64_t rnd_dims[] = {1, 192, (int64_t)synth_frames};
    synth_inputs[5].data = rnd;
    synth_inputs[5].size = 192 * synth_frames;
    synth_inputs[5].shape = tensor_shape_create(rnd_dims, 3);
    synth_inputs[5].dtype = TENSOR_TYPE_FLOAT32;

    // 运行合成器
    TensorData* synth_outputs = NULL;
    size_t synth_num_outputs = 0;

    int64_t synth_start = get_time_ms();
    ret = onnx_session_run_multi(synth_session, synth_inputs, 6, &synth_outputs, &synth_num_outputs);
    int64_t synth_elapsed = get_time_ms() - synth_start;

    // 释放输入形状
    for (int i = 0; i < 6; i++) {
        tensor_shape_free(&synth_inputs[i].shape);
    }

    if (ret != 0 || synth_num_outputs == 0) {
        printf("[FAIL] Synthesizer inference failed\n");
        goto cleanup;
    }

    printf("[PASS] Synthesizer completed in %lld ms\n", (long long)synth_elapsed);
    printf("       Output samples: %zu\n", synth_outputs[0].size);

    // 保存输出音频
    {
        AudioBuffer out_buffer;
        out_buffer.data = (float*)synth_outputs[0].data;
        out_buffer.size = synth_outputs[0].size;
        out_buffer.capacity = synth_outputs[0].size;

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

    // 释放合成器输出
    for (size_t i = 0; i < synth_num_outputs; i++) {
        tensor_data_free(&synth_outputs[i]);
    }
    free(synth_outputs);

cleanup:
    // 清理
    free(phone);
    free(pitch);
    free(pitchf);
    free(rnd);

    for (size_t i = 0; i < cv_num_outputs; i++) {
        tensor_data_free(&cv_outputs[i]);
    }
    free(cv_outputs);

    f0_result_free(&f0_result);
    f0_extractor_destroy(f0_extractor);

    if (audio_16k != input_buffer.data) {
        free(audio_16k);
    }
    audio_buffer_free(&input_buffer);

    onnx_session_destroy(synth_session);
    onnx_session_destroy(contentvec_session);
    onnx_engine_destroy(engine);

    printf("\n========================================\n");
    if (ret == 0) {
        printf("   Voice conversion completed!\n");
    } else {
        printf("   Voice conversion failed!\n");
    }
    printf("========================================\n");

    return ret;
}

int main(int argc, char* argv[]) {
    const char* contentvec_path = NULL;
    const char* synthesizer_path = NULL;
    const char* input_wav = NULL;
    const char* output_wav = "converted_output.wav";
    int speaker_id = 0;
    float pitch_shift = 0.0f;

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
        }
    }

    if (!contentvec_path || !synthesizer_path || !input_wav) {
        printf("RVC Voice Conversion - C++ Implementation\n\n");
        printf("Usage: %s -c <contentvec.onnx> -s <synthesizer.onnx> -i <input.wav> [options]\n\n", argv[0]);
        printf("Required:\n");
        printf("  -c <path>   Path to ContentVec ONNX model\n");
        printf("  -s <path>   Path to Synthesizer ONNX model\n");
        printf("  -i <path>   Path to input WAV file\n");
        printf("\nOptions:\n");
        printf("  -o <path>   Path to output WAV file (default: converted_output.wav)\n");
        printf("  -p <float>  Pitch shift in semitones (default: 0)\n");
        printf("  -sid <int>  Speaker ID (default: 0)\n");
        return 1;
    }

    return rvc_voice_convert(
        contentvec_path,
        synthesizer_path,
        input_wav,
        output_wav,
        speaker_id,
        pitch_shift
    );
}
