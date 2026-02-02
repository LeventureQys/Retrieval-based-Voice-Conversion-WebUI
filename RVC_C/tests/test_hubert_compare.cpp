/**
 * @file test_hubert_compare.cpp
 * @brief 测试 HuBERT/ContentVec 流程，导出每一层的中间结果
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#include "onnx_inference.h"
#include "audio_processor.h"
#include "utils.h"

// 保存数组为二进制文件 (与 Python 兼容)
void save_binary(const char* filepath, const float* data, size_t size) {
    FILE* f = fopen(filepath, "wb");
    if (!f) {
        printf("Failed to open %s for writing\n", filepath);
        return;
    }

    // 写入维度数 (1维)
    uint32_t num_dims = 1;
    fwrite(&num_dims, sizeof(uint32_t), 1, f);

    // 写入维度
    uint64_t dim = size;
    fwrite(&dim, sizeof(uint64_t), 1, f);

    // 写入数据
    fwrite(data, sizeof(float), size, f);

    fclose(f);
    printf("  Saved: %s (%zu elements)\n", filepath, size);
}

void save_binary_3d(const char* filepath, const float* data, size_t d0, size_t d1, size_t d2) {
    FILE* f = fopen(filepath, "wb");
    if (!f) {
        printf("Failed to open %s for writing\n", filepath);
        return;
    }

    // 写入维度数 (3维)
    uint32_t num_dims = 3;
    fwrite(&num_dims, sizeof(uint32_t), 1, f);

    // 写入维度
    uint64_t dims[3] = {d0, d1, d2};
    fwrite(dims, sizeof(uint64_t), 3, f);

    // 写入数据
    fwrite(data, sizeof(float), d0 * d1 * d2, f);

    fclose(f);
    printf("  Saved: %s (%zu x %zu x %zu)\n", filepath, d0, d1, d2);
}

int main(int argc, char* argv[]) {
    const char* contentvec_path = "D:/WorkShop/Retrieval-based-Voice-Conversion-WebUI/RVC_C/test/models/vec-768-layer-12.onnx";
    const char* input_wav = "D:/WorkShop/Retrieval-based-Voice-Conversion-WebUI/RVC_C/test/test_voice/7.wav";
    const char* output_dir = "D:/WorkShop/Retrieval-based-Voice-Conversion-WebUI/RVC_C/test/";

    printf("========================================\n");
    printf("   HuBERT/ContentVec Layer Test\n");
    printf("========================================\n");

    int ret = 0;

    // =========================================================================
    // Step 1: 加载原始音频
    // =========================================================================
    printf("\n[Step 1] Loading original audio...\n");
    AudioBuffer input_buffer = audio_buffer_create(0);
    AudioFormat input_format;

    ret = audio_load_file(input_wav, &input_buffer, &input_format);
    if (ret != 0) {
        printf("  [FAIL] Failed to load audio\n");
        return -1;
    }
    printf("  Original: %zu samples @ %d Hz, %d channels\n",
           input_buffer.size, input_format.sample_rate, input_format.channels);

    // 计算统计信息
    float min_val = input_buffer.data[0], max_val = input_buffer.data[0];
    double sum = 0;
    for (size_t i = 0; i < input_buffer.size; i++) {
        if (input_buffer.data[i] < min_val) min_val = input_buffer.data[i];
        if (input_buffer.data[i] > max_val) max_val = input_buffer.data[i];
        sum += input_buffer.data[i];
    }
    printf("  Range: [%.4f, %.4f]\n", min_val, max_val);
    printf("  Mean: %.6f\n", sum / input_buffer.size);

    // 保存原始音频
    char path[512];
    snprintf(path, sizeof(path), "%scpp_audio_original.bin", output_dir);
    save_binary(path, input_buffer.data, input_buffer.size);

    // =========================================================================
    // Step 2: 重采样到 16kHz
    // =========================================================================
    printf("\n[Step 2] Resampling to 16kHz...\n");
    float* audio_16k = NULL;
    size_t audio_16k_size = 0;

    ret = audio_resample(input_buffer.data, input_buffer.size,
                        input_format.sample_rate, 16000,
                        &audio_16k, &audio_16k_size);
    if (ret != 0) {
        printf("  [FAIL] Resampling failed\n");
        audio_buffer_free(&input_buffer);
        return -1;
    }
    printf("  16kHz: %zu samples\n", audio_16k_size);

    // 计算统计信息
    min_val = audio_16k[0]; max_val = audio_16k[0]; sum = 0;
    for (size_t i = 0; i < audio_16k_size; i++) {
        if (audio_16k[i] < min_val) min_val = audio_16k[i];
        if (audio_16k[i] > max_val) max_val = audio_16k[i];
        sum += audio_16k[i];
    }
    printf("  Range: [%.4f, %.4f]\n", min_val, max_val);
    printf("  Mean: %.6f\n", sum / audio_16k_size);

    // 保存 16kHz 音频
    snprintf(path, sizeof(path), "%scpp_audio_16k.bin", output_dir);
    save_binary(path, audio_16k, audio_16k_size);

    // =========================================================================
    // Step 3: ContentVec 推理
    // =========================================================================
    printf("\n[Step 3] ContentVec inference...\n");

    ONNXEngine* engine = onnx_engine_create(4);
    if (!engine) {
        printf("  [FAIL] Failed to create ONNX engine\n");
        free(audio_16k);
        audio_buffer_free(&input_buffer);
        return -1;
    }

    ONNXSession* cv_session = onnx_session_create(engine, contentvec_path);
    if (!cv_session) {
        printf("  [FAIL] Failed to load ContentVec model\n");
        onnx_engine_destroy(engine);
        free(audio_16k);
        audio_buffer_free(&input_buffer);
        return -1;
    }

    // 准备输入: [1, 1, audio_length]
    int64_t cv_input_dims[] = {1, 1, (int64_t)audio_16k_size};
    TensorData cv_input;
    cv_input.data = audio_16k;
    cv_input.size = audio_16k_size;
    cv_input.shape = tensor_shape_create(cv_input_dims, 3);
    cv_input.dtype = TENSOR_TYPE_FLOAT32;

    printf("  Input shape: [1, 1, %zu]\n", audio_16k_size);
    printf("  Input range: [%.4f, %.4f]\n", min_val, max_val);

    // 运行推理
    TensorData* cv_outputs = NULL;
    size_t cv_num_outputs = 0;

    ret = onnx_session_run_multi(cv_session, &cv_input, 1, &cv_outputs, &cv_num_outputs);
    tensor_shape_free(&cv_input.shape);

    if (ret != 0 || cv_num_outputs == 0) {
        printf("  [FAIL] ContentVec inference failed\n");
        onnx_session_destroy(cv_session);
        onnx_engine_destroy(engine);
        free(audio_16k);
        audio_buffer_free(&input_buffer);
        return -1;
    }

    // ContentVec 输出: [1, frames, 768]
    size_t cv_frames = cv_outputs[0].shape.dims[1];
    size_t cv_dim = cv_outputs[0].shape.dims[2];
    printf("  Output shape: [1, %zu, %zu]\n", cv_frames, cv_dim);

    // 计算统计信息
    float* cv_data = (float*)cv_outputs[0].data;
    min_val = cv_data[0]; max_val = cv_data[0]; sum = 0;
    for (size_t i = 0; i < cv_frames * cv_dim; i++) {
        if (cv_data[i] < min_val) min_val = cv_data[i];
        if (cv_data[i] > max_val) max_val = cv_data[i];
        sum += cv_data[i];
    }
    printf("  Output range: [%.4f, %.4f]\n", min_val, max_val);
    printf("  Output mean: %.6f\n", sum / (cv_frames * cv_dim));

    // 保存 ContentVec 输出
    snprintf(path, sizeof(path), "%scpp_contentvec_output.bin", output_dir);
    save_binary_3d(path, cv_data, 1, cv_frames, cv_dim);

    // =========================================================================
    // Step 4: 特征重复 (2x)
    // =========================================================================
    printf("\n[Step 4] Feature repeat (2x)...\n");
    size_t phone_frames = cv_frames * 2;
    float* phone = (float*)malloc(phone_frames * cv_dim * sizeof(float));

    for (size_t t = 0; t < phone_frames; t++) {
        size_t src_t = t / 2;
        memcpy(&phone[t * cv_dim], &cv_data[src_t * cv_dim], cv_dim * sizeof(float));
    }

    printf("  Phone shape: [1, %zu, %zu]\n", phone_frames, cv_dim);

    // 计算统计信息
    min_val = phone[0]; max_val = phone[0];
    for (size_t i = 0; i < phone_frames * cv_dim; i++) {
        if (phone[i] < min_val) min_val = phone[i];
        if (phone[i] > max_val) max_val = phone[i];
    }
    printf("  Phone range: [%.4f, %.4f]\n", min_val, max_val);

    // 保存 phone
    snprintf(path, sizeof(path), "%scpp_phone.bin", output_dir);
    save_binary_3d(path, phone, 1, phone_frames, cv_dim);

    // =========================================================================
    // 清理
    // =========================================================================
    free(phone);
    for (size_t i = 0; i < cv_num_outputs; i++) {
        tensor_data_free(&cv_outputs[i]);
    }
    free(cv_outputs);
    onnx_session_destroy(cv_session);
    onnx_engine_destroy(engine);
    free(audio_16k);
    audio_buffer_free(&input_buffer);

    printf("\n========================================\n");
    printf("   Test completed!\n");
    printf("========================================\n");

    return 0;
}
