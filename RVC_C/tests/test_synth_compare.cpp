/**
 * @file test_synth_compare.cpp
 * @brief 使用 Python 生成的数据测试合成器，精确对比输出
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#include "onnx_inference.h"
#include "audio_processor.h"

// 读取二进制文件
typedef struct {
    void* data;
    size_t* shape;
    size_t num_dims;
    size_t total_size;
} BinaryData;

BinaryData load_binary(const char* filepath, size_t elem_size) {
    BinaryData result = {NULL, NULL, 0, 0};

    FILE* f = fopen(filepath, "rb");
    if (!f) {
        printf("Failed to open: %s\n", filepath);
        return result;
    }

    // 读取维度数
    uint32_t num_dims;
    fread(&num_dims, sizeof(uint32_t), 1, f);
    result.num_dims = num_dims;

    // 读取每个维度
    result.shape = (size_t*)malloc(num_dims * sizeof(size_t));
    result.total_size = 1;
    for (uint32_t i = 0; i < num_dims; i++) {
        uint64_t dim;
        fread(&dim, sizeof(uint64_t), 1, f);
        result.shape[i] = (size_t)dim;
        result.total_size *= dim;
    }

    // 读取数据
    result.data = malloc(result.total_size * elem_size);
    fread(result.data, elem_size, result.total_size, f);

    fclose(f);
    return result;
}

void free_binary(BinaryData* data) {
    if (data->data) free(data->data);
    if (data->shape) free(data->shape);
    data->data = NULL;
    data->shape = NULL;
}

int main(int argc, char* argv[]) {
    const char* model_path = "D:/WorkShop/Retrieval-based-Voice-Conversion-WebUI/RVC_C/test/models/Rem_e440_s38720.onnx";
    const char* data_dir = "D:/WorkShop/Retrieval-based-Voice-Conversion-WebUI/RVC_C/test/";

    printf("========================================\n");
    printf("   Synthesizer Comparison Test\n");
    printf("========================================\n");

    // 加载 Python 生成的数据
    printf("\n[1] Loading Python-generated data...\n");

    char path[512];

    snprintf(path, sizeof(path), "%sbin_phone.bin", data_dir);
    BinaryData phone_data = load_binary(path, sizeof(float));
    printf("    phone: %zu x %zu x %zu\n", phone_data.shape[0], phone_data.shape[1], phone_data.shape[2]);

    snprintf(path, sizeof(path), "%sbin_pitch.bin", data_dir);
    BinaryData pitch_data = load_binary(path, sizeof(int64_t));
    printf("    pitch: %zu x %zu\n", pitch_data.shape[0], pitch_data.shape[1]);

    snprintf(path, sizeof(path), "%sbin_pitchf.bin", data_dir);
    BinaryData pitchf_data = load_binary(path, sizeof(float));
    printf("    pitchf: %zu x %zu\n", pitchf_data.shape[0], pitchf_data.shape[1]);

    snprintf(path, sizeof(path), "%sbin_rnd.bin", data_dir);
    BinaryData rnd_data = load_binary(path, sizeof(float));
    printf("    rnd: %zu x %zu x %zu\n", rnd_data.shape[0], rnd_data.shape[1], rnd_data.shape[2]);

    snprintf(path, sizeof(path), "%sbin_output_python.bin", data_dir);
    BinaryData python_output = load_binary(path, sizeof(float));
    printf("    python_output: %zu samples\n", python_output.total_size);

    // 创建 ONNX 引擎
    printf("\n[2] Loading ONNX model...\n");
    ONNXEngine* engine = onnx_engine_create(4);
    ONNXSession* session = onnx_session_create(engine, model_path);
    if (!session) {
        printf("Failed to load model\n");
        return -1;
    }
    printf("    Model loaded.\n");

    // 准备输入
    printf("\n[3] Running inference...\n");

    size_t synth_frames = phone_data.shape[1];
    int64_t phone_lengths_val = (int64_t)synth_frames;
    int64_t ds_val = 0;

    TensorData inputs[6];

    // phone
    int64_t phone_dims[] = {1, (int64_t)phone_data.shape[1], (int64_t)phone_data.shape[2]};
    inputs[0].data = phone_data.data;
    inputs[0].size = phone_data.total_size;
    inputs[0].shape = tensor_shape_create(phone_dims, 3);
    inputs[0].dtype = TENSOR_TYPE_FLOAT32;

    // phone_lengths
    int64_t phone_lengths_dims[] = {1};
    inputs[1].data = &phone_lengths_val;
    inputs[1].size = 1;
    inputs[1].shape = tensor_shape_create(phone_lengths_dims, 1);
    inputs[1].dtype = TENSOR_TYPE_INT64;

    // pitch
    int64_t pitch_dims[] = {1, (int64_t)pitch_data.shape[1]};
    inputs[2].data = pitch_data.data;
    inputs[2].size = pitch_data.total_size;
    inputs[2].shape = tensor_shape_create(pitch_dims, 2);
    inputs[2].dtype = TENSOR_TYPE_INT64;

    // pitchf
    int64_t pitchf_dims[] = {1, (int64_t)pitchf_data.shape[1]};
    inputs[3].data = pitchf_data.data;
    inputs[3].size = pitchf_data.total_size;
    inputs[3].shape = tensor_shape_create(pitchf_dims, 2);
    inputs[3].dtype = TENSOR_TYPE_FLOAT32;

    // ds
    int64_t ds_dims[] = {1};
    inputs[4].data = &ds_val;
    inputs[4].size = 1;
    inputs[4].shape = tensor_shape_create(ds_dims, 1);
    inputs[4].dtype = TENSOR_TYPE_INT64;

    // rnd
    int64_t rnd_dims[] = {1, (int64_t)rnd_data.shape[1], (int64_t)rnd_data.shape[2]};
    inputs[5].data = rnd_data.data;
    inputs[5].size = rnd_data.total_size;
    inputs[5].shape = tensor_shape_create(rnd_dims, 3);
    inputs[5].dtype = TENSOR_TYPE_FLOAT32;

    // 运行推理
    TensorData* outputs = NULL;
    size_t num_outputs = 0;

    int ret = onnx_session_run_multi(session, inputs, 6, &outputs, &num_outputs);

    // 释放输入形状
    for (int i = 0; i < 6; i++) {
        tensor_shape_free(&inputs[i].shape);
    }

    if (ret != 0) {
        printf("    Inference failed!\n");
        return -1;
    }

    printf("    Output samples: %zu\n", outputs[0].size);

    // 对比输出
    printf("\n[4] Comparing outputs...\n");

    float* cpp_out = (float*)outputs[0].data;
    float* py_out = (float*)python_output.data;
    size_t min_len = outputs[0].size < python_output.total_size ? outputs[0].size : python_output.total_size;

    double max_diff = 0;
    double sum_diff = 0;
    size_t diff_count_001 = 0;
    size_t diff_count_01 = 0;

    for (size_t i = 0; i < min_len; i++) {
        double diff = fabs(cpp_out[i] - py_out[i]);
        if (diff > max_diff) max_diff = diff;
        sum_diff += diff;
        if (diff > 0.01) diff_count_001++;
        if (diff > 0.1) diff_count_01++;
    }

    printf("    Max diff: %.6f\n", max_diff);
    printf("    Mean diff: %.6f\n", sum_diff / min_len);
    printf("    Samples with diff > 0.01: %zu (%.1f%%)\n", diff_count_001, 100.0 * diff_count_001 / min_len);
    printf("    Samples with diff > 0.1: %zu (%.1f%%)\n", diff_count_01, 100.0 * diff_count_01 / min_len);

    // 保存 C++ 输出
    AudioBuffer out_buffer;
    out_buffer.data = cpp_out;
    out_buffer.size = outputs[0].size;
    out_buffer.capacity = outputs[0].size;

    AudioFormat out_format;
    out_format.sample_rate = 48000;
    out_format.channels = 1;
    out_format.bits_per_sample = 32;

    snprintf(path, sizeof(path), "%scpp_with_python_data.wav", data_dir);
    audio_save_file(path, &out_buffer, &out_format);
    printf("\n    Saved: %s\n", path);

    // 清理
    for (size_t i = 0; i < num_outputs; i++) {
        tensor_data_free(&outputs[i]);
    }
    free(outputs);

    free_binary(&phone_data);
    free_binary(&pitch_data);
    free_binary(&pitchf_data);
    free_binary(&rnd_data);
    free_binary(&python_output);

    onnx_session_destroy(session);
    onnx_engine_destroy(engine);

    printf("\n========================================\n");
    printf("   Test completed!\n");
    printf("========================================\n");

    return 0;
}
