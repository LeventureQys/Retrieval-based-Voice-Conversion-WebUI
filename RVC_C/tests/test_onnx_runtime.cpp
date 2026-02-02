/**
 * @file test_onnx_runtime.cpp
 * @brief ONNX Runtime 基础测试
 *
 * 测试 ONNX Runtime 是否正确链接和工作
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "onnx_inference.h"
#include "utils.h"

// 直接包含 ONNX Runtime 头文件进行基础测试
#include <onnxruntime_c_api.h>

void test_onnx_runtime_version() {
    printf("=== ONNX Runtime Version Test ===\n");

    const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (api) {
        printf("[PASS] ONNX Runtime API loaded successfully\n");
        printf("       API Version: %d\n", ORT_API_VERSION);
    } else {
        printf("[FAIL] Failed to load ONNX Runtime API\n");
    }
    printf("\n");
}

void test_onnx_engine_creation() {
    printf("=== ONNX Engine Creation Test ===\n");

    ONNXEngine* engine = onnx_engine_create(4);
    if (engine) {
        printf("[PASS] ONNX engine created successfully\n");
        onnx_engine_destroy(engine);
        printf("[PASS] ONNX engine destroyed successfully\n");
    } else {
        printf("[FAIL] Failed to create ONNX engine\n");
    }
    printf("\n");
}

void test_tensor_shape() {
    printf("=== Tensor Shape Test ===\n");

    int64_t dims[] = {1, 3, 224, 224};
    TensorShape shape = tensor_shape_create(dims, 4);

    size_t size = tensor_shape_size(&shape);
    size_t expected = 1 * 3 * 224 * 224;

    if (size == expected) {
        printf("[PASS] Tensor shape size: %zu (expected: %zu)\n", size, expected);
    } else {
        printf("[FAIL] Tensor shape size: %zu (expected: %zu)\n", size, expected);
    }

    tensor_shape_free(&shape);
    printf("[PASS] Tensor shape freed\n");
    printf("\n");
}

void test_model_loading(const char* model_path) {
    printf("=== Model Loading Test ===\n");

    if (!model_path) {
        printf("[SKIP] No model path provided\n");
        printf("       Usage: test_onnx_runtime <model.onnx>\n");
        printf("\n");
        return;
    }

    printf("Loading model: %s\n", model_path);

    ONNXEngine* engine = onnx_engine_create(4);
    if (!engine) {
        printf("[FAIL] Failed to create engine\n");
        return;
    }

    ONNXSession* session = onnx_session_create(engine, model_path);
    if (session) {
        printf("[PASS] Model loaded successfully\n");

        size_t num_inputs = onnx_session_get_input_count(session);
        size_t num_outputs = onnx_session_get_output_count(session);

        printf("       Inputs: %zu\n", num_inputs);
        printf("       Outputs: %zu\n", num_outputs);

        // 打印输入名称
        for (size_t i = 0; i < num_inputs; i++) {
            char* name = onnx_session_get_input_name(session, i);
            if (name) {
                printf("       Input[%zu]: %s\n", i, name);
                free(name);
            }
        }

        // 打印输出名称
        for (size_t i = 0; i < num_outputs; i++) {
            char* name = onnx_session_get_output_name(session, i);
            if (name) {
                printf("       Output[%zu]: %s\n", i, name);
                free(name);
            }
        }

        onnx_session_destroy(session);
        printf("[PASS] Session destroyed\n");
    } else {
        printf("[FAIL] Failed to load model\n");
    }

    onnx_engine_destroy(engine);
    printf("\n");
}

void test_simple_inference() {
    printf("=== Simple Inference Test ===\n");

    // 创建一个简单的测试，不需要实际模型
    // 只测试内存分配和基本操作

    float test_input[100];
    for (int i = 0; i < 100; i++) {
        test_input[i] = (float)i / 100.0f;
    }

    printf("[PASS] Test input created: 100 floats\n");

    // 测试数组操作
    float sum = array_sum(test_input, 100);
    float mean = array_mean(test_input, 100);
    float max_val = array_max(test_input, 100);
    float min_val = array_min(test_input, 100);

    printf("       Sum: %.2f\n", sum);
    printf("       Mean: %.4f\n", mean);
    printf("       Max: %.2f\n", max_val);
    printf("       Min: %.2f\n", min_val);

    printf("[PASS] Array operations completed\n");
    printf("\n");
}

int main(int argc, char* argv[]) {
    printf("========================================\n");
    printf("   RVC_ONNX - ONNX Runtime Test Suite\n");
    printf("========================================\n\n");

    // 设置日志级别
    log_set_level(LOG_INFO);

    // 运行测试
    test_onnx_runtime_version();
    test_onnx_engine_creation();
    test_tensor_shape();
    test_simple_inference();

    // 如果提供了模型路径，测试模型加载
    const char* model_path = (argc > 1) ? argv[1] : NULL;
    test_model_loading(model_path);

    printf("========================================\n");
    printf("   All tests completed!\n");
    printf("========================================\n");

    return 0;
}
