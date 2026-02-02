/**
 * @file onnx_inference.cpp
 * @brief ONNX Runtime 推理引擎实现
 */

#include "onnx_inference.h"
#include "utils.h"
#include <onnxruntime_cxx_api.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <string>

#ifdef _WIN32
#include <windows.h>
// Windows 下将 UTF-8 路径转换为宽字符
static std::wstring utf8_to_wstring(const char* utf8_str) {
    if (!utf8_str) return L"";
    int len = MultiByteToWideChar(CP_UTF8, 0, utf8_str, -1, nullptr, 0);
    if (len <= 0) return L"";
    std::wstring wstr(len, 0);
    MultiByteToWideChar(CP_UTF8, 0, utf8_str, -1, &wstr[0], len);
    wstr.resize(len - 1); // 移除末尾的 null 字符
    return wstr;
}
#endif

// =============================================================================
// 内部结构定义
// =============================================================================

struct ONNXEngine {
    Ort::Env env;
    Ort::SessionOptions session_options;
    int num_threads;

    ONNXEngine(int threads)
        : env(ORT_LOGGING_LEVEL_WARNING, "RVC_ONNX")
        , num_threads(threads) {
        session_options.SetIntraOpNumThreads(threads);
        session_options.SetInterOpNumThreads(threads);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    }
};

struct ONNXSession {
    ONNXEngine* engine;
    Ort::Session session;
    Ort::AllocatorWithDefaultOptions allocator;

    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    std::vector<Ort::AllocatedStringPtr> input_name_ptrs;
    std::vector<Ort::AllocatedStringPtr> output_name_ptrs;

    ONNXSession(ONNXEngine* eng, const char* model_path)
        : engine(eng)
#ifdef _WIN32
        , session(eng->env, utf8_to_wstring(model_path).c_str(), eng->session_options)
#else
        , session(eng->env, model_path, eng->session_options)
#endif
    {

        // 获取输入名称
        size_t num_inputs = session.GetInputCount();
        for (size_t i = 0; i < num_inputs; i++) {
            auto name_ptr = session.GetInputNameAllocated(i, allocator);
            input_names.push_back(name_ptr.get());
            input_name_ptrs.push_back(std::move(name_ptr));
        }

        // 获取输出名称
        size_t num_outputs = session.GetOutputCount();
        for (size_t i = 0; i < num_outputs; i++) {
            auto name_ptr = session.GetOutputNameAllocated(i, allocator);
            output_names.push_back(name_ptr.get());
            output_name_ptrs.push_back(std::move(name_ptr));
        }
    }
};

// =============================================================================
// 引擎管理
// =============================================================================

extern "C" {

ONNXEngine* onnx_engine_create(int num_threads) {
    try {
        return new ONNXEngine(num_threads);
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to create ONNX engine: %s", e.what());
        return nullptr;
    }
}

void onnx_engine_destroy(ONNXEngine* engine) {
    if (engine) {
        delete engine;
    }
}

// =============================================================================
// 会话管理
// =============================================================================

ONNXSession* onnx_session_create(ONNXEngine* engine, const char* model_path) {
    if (!engine || !model_path) {
        LOG_ERROR("Invalid parameters for session creation");
        return nullptr;
    }

    try {
        LOG_INFO("Loading ONNX model: %s", model_path);
        ONNXSession* session = new ONNXSession(engine, model_path);
        LOG_INFO("Model loaded successfully. Inputs: %zu, Outputs: %zu",
                 session->input_names.size(), session->output_names.size());
        return session;
    } catch (const Ort::Exception& e) {
        LOG_ERROR("ONNX Runtime error: %s", e.what());
        return nullptr;
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to load model: %s", e.what());
        return nullptr;
    }
}

void onnx_session_destroy(ONNXSession* session) {
    if (session) {
        delete session;
    }
}

size_t onnx_session_get_input_count(ONNXSession* session) {
    return session ? session->input_names.size() : 0;
}

size_t onnx_session_get_output_count(ONNXSession* session) {
    return session ? session->output_names.size() : 0;
}

char* onnx_session_get_input_name(ONNXSession* session, size_t index) {
    if (!session || index >= session->input_names.size()) {
        return nullptr;
    }
    return strdup(session->input_names[index].c_str());
}

char* onnx_session_get_output_name(ONNXSession* session, size_t index) {
    if (!session || index >= session->output_names.size()) {
        return nullptr;
    }
    return strdup(session->output_names[index].c_str());
}

// =============================================================================
// 推理执行
// =============================================================================

int onnx_session_run_single(
    ONNXSession* session,
    const float* input,
    const TensorShape* input_shape,
    float** output,
    TensorShape* output_shape
) {
    if (!session || !input || !input_shape || !output || !output_shape) {
        return -1;
    }

    try {
        // 创建内存信息
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
            OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

        // 计算输入大小
        size_t input_size = tensor_shape_size(input_shape);

        // 创建输入张量
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            const_cast<float*>(input),
            input_size,
            input_shape->dims,
            input_shape->num_dims
        );

        // 准备输入/输出名称
        const char* input_names[] = { session->input_names[0].c_str() };
        const char* output_names[] = { session->output_names[0].c_str() };

        // 运行推理
        auto output_tensors = session->session.Run(
            Ort::RunOptions{nullptr},
            input_names, &input_tensor, 1,
            output_names, 1
        );

        // 获取输出
        auto& output_tensor = output_tensors[0];
        auto type_info = output_tensor.GetTensorTypeAndShapeInfo();
        auto shape = type_info.GetShape();
        size_t output_size = type_info.GetElementCount();

        // 分配输出内存
        *output = (float*)malloc(output_size * sizeof(float));
        if (!*output) {
            return -2;
        }

        // 复制输出数据
        float* output_data = output_tensor.GetTensorMutableData<float>();
        memcpy(*output, output_data, output_size * sizeof(float));

        // 设置输出形状
        output_shape->num_dims = shape.size();
        output_shape->dims = (int64_t*)malloc(shape.size() * sizeof(int64_t));
        for (size_t i = 0; i < shape.size(); i++) {
            output_shape->dims[i] = shape[i];
        }

        return 0;
    } catch (const Ort::Exception& e) {
        LOG_ERROR("ONNX inference error: %s", e.what());
        return -3;
    } catch (const std::exception& e) {
        LOG_ERROR("Inference error: %s", e.what());
        return -4;
    }
}

int onnx_session_run_multi(
    ONNXSession* session,
    const TensorData* inputs,
    size_t num_inputs,
    TensorData** outputs,
    size_t* num_outputs
) {
    if (!session || !inputs || !outputs || !num_outputs) {
        return -1;
    }

    try {
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
            OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

        // 创建输入张量
        std::vector<Ort::Value> input_tensors;
        std::vector<const char*> input_names;

        for (size_t i = 0; i < num_inputs; i++) {
            // 根据数据类型创建不同类型的张量
            switch (inputs[i].dtype) {
                case TENSOR_TYPE_INT64:
                    input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
                        memory_info,
                        (int64_t*)inputs[i].data,
                        inputs[i].size,
                        inputs[i].shape.dims,
                        inputs[i].shape.num_dims
                    ));
                    break;
                case TENSOR_TYPE_INT32:
                    input_tensors.push_back(Ort::Value::CreateTensor<int32_t>(
                        memory_info,
                        (int32_t*)inputs[i].data,
                        inputs[i].size,
                        inputs[i].shape.dims,
                        inputs[i].shape.num_dims
                    ));
                    break;
                case TENSOR_TYPE_FLOAT32:
                default:
                    input_tensors.push_back(Ort::Value::CreateTensor<float>(
                        memory_info,
                        (float*)inputs[i].data,
                        inputs[i].size,
                        inputs[i].shape.dims,
                        inputs[i].shape.num_dims
                    ));
                    break;
            }
            input_names.push_back(session->input_names[i].c_str());
        }

        // 准备输出名称
        std::vector<const char*> output_names;
        for (const auto& name : session->output_names) {
            output_names.push_back(name.c_str());
        }

        // 运行推理
        auto output_tensors = session->session.Run(
            Ort::RunOptions{nullptr},
            input_names.data(), input_tensors.data(), num_inputs,
            output_names.data(), output_names.size()
        );

        // 分配输出
        *num_outputs = output_tensors.size();
        *outputs = (TensorData*)malloc(*num_outputs * sizeof(TensorData));

        for (size_t i = 0; i < *num_outputs; i++) {
            auto& tensor = output_tensors[i];
            auto type_info = tensor.GetTensorTypeAndShapeInfo();
            auto shape = type_info.GetShape();
            size_t size = type_info.GetElementCount();

            (*outputs)[i].size = size;
            (*outputs)[i].data = malloc(size * sizeof(float));
            (*outputs)[i].shape.num_dims = shape.size();
            (*outputs)[i].shape.dims = (int64_t*)malloc(shape.size() * sizeof(int64_t));
            (*outputs)[i].dtype = TENSOR_TYPE_FLOAT32;

            memcpy((*outputs)[i].data, tensor.GetTensorMutableData<float>(), size * sizeof(float));
            for (size_t j = 0; j < shape.size(); j++) {
                (*outputs)[i].shape.dims[j] = shape[j];
            }
        }

        return 0;
    } catch (const Ort::Exception& e) {
        LOG_ERROR("ONNX inference error: %s", e.what());
        return -3;
    }
}

// =============================================================================
// 工具函数
// =============================================================================

void tensor_data_free(TensorData* tensor) {
    if (tensor) {
        if (tensor->data) {
            free(tensor->data);
            tensor->data = nullptr;
        }
        tensor_shape_free(&tensor->shape);
        tensor->size = 0;
    }
}

TensorShape tensor_shape_create(const int64_t* dims, size_t num_dims) {
    TensorShape shape;
    shape.num_dims = num_dims;
    shape.dims = (int64_t*)malloc(num_dims * sizeof(int64_t));
    if (shape.dims && dims) {
        memcpy(shape.dims, dims, num_dims * sizeof(int64_t));
    }
    return shape;
}

void tensor_shape_free(TensorShape* shape) {
    if (shape && shape->dims) {
        free(shape->dims);
        shape->dims = nullptr;
        shape->num_dims = 0;
    }
}

size_t tensor_shape_size(const TensorShape* shape) {
    if (!shape || !shape->dims || shape->num_dims == 0) {
        return 0;
    }
    size_t size = 1;
    for (size_t i = 0; i < shape->num_dims; i++) {
        size *= (size_t)shape->dims[i];
    }
    return size;
}

} // extern "C"
