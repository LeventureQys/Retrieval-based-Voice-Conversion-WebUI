/**
 * @file test_synthesizer.cpp
 * @brief 合成器 ONNX 模型推理测试
 *
 * 测试直接使用合成器模型进行推理
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "onnx_inference.h"
#include "audio_processor.h"
#include "f0_extractor.h"
#include "utils.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// 生成测试用的 phone 特征 (模拟 HuBERT 输出)
void generate_test_phone_features(float* phone, int time_steps, int hidden_channels) {
    // 生成随机特征，模拟 HuBERT 输出
    for (int t = 0; t < time_steps; t++) {
        for (int c = 0; c < hidden_channels; c++) {
            // 使用简单的正弦波模式
            phone[t * hidden_channels + c] = 0.1f * sinf((float)(t * c) * 0.01f);
        }
    }
}

// 生成测试用的 pitch 数据
void generate_test_pitch(int64_t* pitch, float* pitchf, int time_steps, float base_freq) {
    for (int t = 0; t < time_steps; t++) {
        // 生成平稳的音高
        float freq = base_freq + 10.0f * sinf((float)t * 0.05f);

        // 量化到 1-255 范围
        int pitch_val = (int)(12.0f * log2f(freq / 10.0f));
        if (pitch_val < 1) pitch_val = 1;
        if (pitch_val > 255) pitch_val = 255;

        pitch[t] = pitch_val;
        pitchf[t] = freq;
    }
}

int test_synthesizer_inference(const char* model_path) {
    printf("\n=== Synthesizer Inference Test ===\n");

    // 创建 ONNX 引擎
    ONNXEngine* engine = onnx_engine_create(4);
    if (!engine) {
        printf("[FAIL] Failed to create ONNX engine\n");
        return -1;
    }
    printf("[PASS] ONNX engine created\n");

    // 加载合成器模型
    ONNXSession* session = onnx_session_create(engine, model_path);
    if (!session) {
        printf("[FAIL] Failed to load synthesizer model\n");
        onnx_engine_destroy(engine);
        return -1;
    }
    printf("[PASS] Synthesizer model loaded\n");

    // 获取模型信息
    size_t num_inputs = onnx_session_get_input_count(session);
    size_t num_outputs = onnx_session_get_output_count(session);
    printf("       Inputs: %zu, Outputs: %zu\n", num_inputs, num_outputs);

    // 打印输入名称
    for (size_t i = 0; i < num_inputs; i++) {
        char* name = onnx_session_get_input_name(session, i);
        printf("       Input[%zu]: %s\n", i, name);
        free(name);
    }

    // 准备测试数据
    // RVC 合成器输入:
    // - phone: [1, time_steps, 768] - HuBERT 特征
    // - phone_lengths: [1] - 序列长度
    // - pitch: [1, time_steps] - 量化音高 (1-255)
    // - pitchf: [1, time_steps] - 连续音高 (Hz)
    // - ds: [1] - 说话人 ID
    // - rnd: [1, 192, time_steps] - 随机噪声

    const int time_steps = 100;  // 约 0.5 秒
    const int hidden_channels = 768;  // v2 模型
    const int latent_channels = 192;

    // 分配内存
    float* phone = (float*)malloc(time_steps * hidden_channels * sizeof(float));
    int64_t phone_lengths_val = time_steps;
    int64_t* pitch = (int64_t*)malloc(time_steps * sizeof(int64_t));
    float* pitchf = (float*)malloc(time_steps * sizeof(float));
    int64_t ds_val = 0;  // 说话人 ID
    float* rnd = (float*)malloc(latent_channels * time_steps * sizeof(float));

    if (!phone || !pitch || !pitchf || !rnd) {
        printf("[FAIL] Failed to allocate test data\n");
        onnx_session_destroy(session);
        onnx_engine_destroy(engine);
        return -1;
    }

    // 生成测试数据
    generate_test_phone_features(phone, time_steps, hidden_channels);
    generate_test_pitch(pitch, pitchf, time_steps, 220.0f);  // A3 = 220 Hz

    // 生成随机噪声
    for (int i = 0; i < latent_channels * time_steps; i++) {
        rnd[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    }

    printf("[PASS] Test data generated\n");
    printf("       Time steps: %d\n", time_steps);
    printf("       Hidden channels: %d\n", hidden_channels);

    // 准备输入张量
    TensorData inputs[6];

    // phone: [1, time_steps, hidden_channels] - float32
    int64_t phone_dims[] = {1, time_steps, hidden_channels};
    inputs[0].data = phone;
    inputs[0].size = time_steps * hidden_channels;
    inputs[0].shape = tensor_shape_create(phone_dims, 3);
    inputs[0].dtype = TENSOR_TYPE_FLOAT32;

    // phone_lengths: [1] - int64
    int64_t phone_lengths_dims[] = {1};
    inputs[1].data = &phone_lengths_val;
    inputs[1].size = 1;
    inputs[1].shape = tensor_shape_create(phone_lengths_dims, 1);
    inputs[1].dtype = TENSOR_TYPE_INT64;

    // pitch: [1, time_steps] - int64
    int64_t pitch_dims[] = {1, time_steps};
    inputs[2].data = pitch;
    inputs[2].size = time_steps;
    inputs[2].shape = tensor_shape_create(pitch_dims, 2);
    inputs[2].dtype = TENSOR_TYPE_INT64;

    // pitchf: [1, time_steps] - float32
    int64_t pitchf_dims[] = {1, time_steps};
    inputs[3].data = pitchf;
    inputs[3].size = time_steps;
    inputs[3].shape = tensor_shape_create(pitchf_dims, 2);
    inputs[3].dtype = TENSOR_TYPE_FLOAT32;

    // ds: [1] - int64
    int64_t ds_dims[] = {1};
    inputs[4].data = &ds_val;
    inputs[4].size = 1;
    inputs[4].shape = tensor_shape_create(ds_dims, 1);
    inputs[4].dtype = TENSOR_TYPE_INT64;

    // rnd: [1, 192, time_steps] - float32
    int64_t rnd_dims[] = {1, latent_channels, time_steps};
    inputs[5].data = rnd;
    inputs[5].size = latent_channels * time_steps;
    inputs[5].shape = tensor_shape_create(rnd_dims, 3);
    inputs[5].dtype = TENSOR_TYPE_FLOAT32;

    printf("[INFO] Running inference...\n");

    // 运行推理
    TensorData* outputs = NULL;
    size_t num_outputs_result = 0;

    int64_t start_time = get_time_ms();
    int ret = onnx_session_run_multi(session, inputs, 6, &outputs, &num_outputs_result);
    int64_t elapsed = get_time_ms() - start_time;

    if (ret != 0) {
        printf("[FAIL] Inference failed with error: %d\n", ret);
        printf("       Note: This may be due to input type mismatch.\n");
        printf("       The model expects specific tensor types (int64 for pitch, ds, phone_lengths)\n");
    } else {
        printf("[PASS] Inference completed in %lld ms\n", (long long)elapsed);
        printf("       Output tensors: %zu\n", num_outputs_result);

        if (outputs && num_outputs_result > 0) {
            printf("       Output size: %zu\n", outputs[0].size);
            printf("       Output shape dims: %zu\n", outputs[0].shape.num_dims);

            // 计算输出统计
            float* output_data = (float*)outputs[0].data;
            float min_val = output_data[0];
            float max_val = output_data[0];
            float sum = 0.0f;

            for (size_t i = 0; i < outputs[0].size; i++) {
                float val = output_data[i];
                if (val < min_val) min_val = val;
                if (val > max_val) max_val = val;
                sum += val;
            }

            printf("       Output min: %.4f, max: %.4f, mean: %.4f\n",
                   min_val, max_val, sum / outputs[0].size);

            // 保存输出音频
            AudioBuffer out_buffer;
            out_buffer.data = output_data;
            out_buffer.size = outputs[0].size;
            out_buffer.capacity = outputs[0].size;

            AudioFormat out_format;
            out_format.sample_rate = 48000;
            out_format.channels = 1;
            out_format.bits_per_sample = 32;

            const char* output_path = "cpp_synth_output.wav";
            if (audio_save_file(output_path, &out_buffer, &out_format) == 0) {
                printf("[PASS] Saved output audio to: %s\n", output_path);
            }

            // 释放输出
            for (size_t i = 0; i < num_outputs_result; i++) {
                tensor_data_free(&outputs[i]);
            }
            free(outputs);
        }
    }

    // 清理
    for (int i = 0; i < 6; i++) {
        tensor_shape_free(&inputs[i].shape);
    }
    free(phone);
    free(pitch);
    free(pitchf);
    free(rnd);

    onnx_session_destroy(session);
    onnx_engine_destroy(engine);

    return ret;
}

int test_audio_to_audio(const char* model_path, const char* input_wav, const char* output_wav) {
    printf("\n=== Audio-to-Audio Test ===\n");

    // 加载输入音频
    AudioBuffer input_buffer = audio_buffer_create(0);
    AudioFormat format;

    int ret = audio_load_file(input_wav, &input_buffer, &format);
    if (ret != 0) {
        printf("[FAIL] Failed to load input audio: %s\n", input_wav);
        return -1;
    }
    printf("[PASS] Loaded input audio: %zu samples, %d Hz\n",
           input_buffer.size, format.sample_rate);

    // 创建 F0 提取器
    F0Extractor* f0_extractor = f0_extractor_create(F0_METHOD_HARVEST, format.sample_rate);
    if (!f0_extractor) {
        printf("[FAIL] Failed to create F0 extractor\n");
        audio_buffer_free(&input_buffer);
        return -1;
    }

    // 提取 F0
    F0Result f0_result;
    memset(&f0_result, 0, sizeof(F0Result));

    ret = f0_extract_float(f0_extractor, input_buffer.data, input_buffer.size, &f0_result);
    if (ret != 0) {
        printf("[FAIL] F0 extraction failed\n");
        f0_extractor_destroy(f0_extractor);
        audio_buffer_free(&input_buffer);
        return -1;
    }
    printf("[PASS] F0 extracted: %zu frames\n", f0_result.length);

    // 计算 F0 统计
    float f0_sum = 0.0f;
    int voiced_count = 0;
    for (size_t i = 0; i < f0_result.length; i++) {
        if (f0_result.f0[i] > 0) {
            f0_sum += (float)f0_result.f0[i];
            voiced_count++;
        }
    }
    if (voiced_count > 0) {
        printf("       Average F0: %.1f Hz (voiced frames: %d)\n",
               f0_sum / voiced_count, voiced_count);
    }

    // 清理
    f0_result_free(&f0_result);
    f0_extractor_destroy(f0_extractor);

    // 目前只是复制输入到输出（完整实现需要 HuBERT 模型）
    printf("[INFO] Full pipeline requires HuBERT model for feature extraction\n");
    printf("[INFO] Copying input to output as placeholder\n");

    // 保存输出
    ret = audio_save_file(output_wav, &input_buffer, &format);
    if (ret != 0) {
        printf("[FAIL] Failed to save output audio\n");
        audio_buffer_free(&input_buffer);
        return -1;
    }
    printf("[PASS] Saved output audio: %s\n", output_wav);

    audio_buffer_free(&input_buffer);
    return 0;
}

int main(int argc, char* argv[]) {
    printf("========================================\n");
    printf("   RVC_ONNX - Synthesizer Test Suite\n");
    printf("========================================\n");

    const char* model_path = NULL;
    const char* input_wav = NULL;
    const char* output_wav = "output.wav";

    // 解析命令行参数
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            input_wav = argv[++i];
        } else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            output_wav = argv[++i];
        }
    }

    if (!model_path) {
        printf("\nUsage: %s -m <model.onnx> [-i <input.wav>] [-o <output.wav>]\n", argv[0]);
        printf("\nOptions:\n");
        printf("  -m <path>  Path to synthesizer ONNX model\n");
        printf("  -i <path>  Path to input WAV file (optional)\n");
        printf("  -o <path>  Path to output WAV file (default: output.wav)\n");
        return 1;
    }

    // 测试合成器推理
    int ret = test_synthesizer_inference(model_path);

    // 如果提供了输入音频，进行音频到音频测试
    if (input_wav) {
        ret = test_audio_to_audio(model_path, input_wav, output_wav);
    }

    printf("\n========================================\n");
    printf("   Tests completed!\n");
    printf("========================================\n");

    return ret;
}
