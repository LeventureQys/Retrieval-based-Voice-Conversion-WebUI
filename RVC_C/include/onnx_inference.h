/**
 * @file onnx_inference.h
 * @brief ONNX Runtime 推理引擎封装
 */

#ifndef ONNX_INFERENCE_H
#define ONNX_INFERENCE_H

#include <onnxruntime_c_api.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// 类型定义
// =============================================================================

/** ONNX 推理引擎句柄 */
typedef struct ONNXEngine ONNXEngine;

/** ONNX 会话句柄 */
typedef struct ONNXSession ONNXSession;

/** 张量形状 */
typedef struct {
    int64_t* dims;
    size_t num_dims;
} TensorShape;

/** 张量数据类型 */
typedef enum {
    TENSOR_TYPE_FLOAT32 = 0,
    TENSOR_TYPE_INT64 = 1,
    TENSOR_TYPE_INT32 = 2,
} TensorDataType;

/** 张量数据 */
typedef struct {
    void* data;           // 数据指针 (可以是 float* 或 int64_t*)
    size_t size;          // 元素数量
    TensorShape shape;
    TensorDataType dtype; // 数据类型
} TensorData;

/** 向后兼容的 float 张量数据 (已废弃，请使用 TensorData) */
#define TensorDataFloat TensorData

// =============================================================================
// 引擎管理
// =============================================================================

/**
 * @brief 创建ONNX推理引擎
 * @param num_threads 推理线程数
 * @return 引擎句柄，失败返回NULL
 */
ONNXEngine* onnx_engine_create(int num_threads);

/**
 * @brief 销毁ONNX推理引擎
 * @param engine 引擎句柄
 */
void onnx_engine_destroy(ONNXEngine* engine);

// =============================================================================
// 会话管理
// =============================================================================

/**
 * @brief 加载ONNX模型创建会话
 * @param engine 引擎句柄
 * @param model_path 模型文件路径
 * @return 会话句柄，失败返回NULL
 */
ONNXSession* onnx_session_create(ONNXEngine* engine, const char* model_path);

/**
 * @brief 销毁ONNX会话
 * @param session 会话句柄
 */
void onnx_session_destroy(ONNXSession* session);

/**
 * @brief 获取模型输入数量
 * @param session 会话句柄
 * @return 输入数量
 */
size_t onnx_session_get_input_count(ONNXSession* session);

/**
 * @brief 获取模型输出数量
 * @param session 会话句柄
 * @return 输出数量
 */
size_t onnx_session_get_output_count(ONNXSession* session);

/**
 * @brief 获取输入名称
 * @param session 会话句柄
 * @param index 输入索引
 * @return 输入名称 (需要调用者释放)
 */
char* onnx_session_get_input_name(ONNXSession* session, size_t index);

/**
 * @brief 获取输出名称
 * @param session 会话句柄
 * @param index 输出索引
 * @return 输出名称 (需要调用者释放)
 */
char* onnx_session_get_output_name(ONNXSession* session, size_t index);

// =============================================================================
// 推理执行
// =============================================================================

/**
 * @brief 执行单输入单输出推理
 * @param session 会话句柄
 * @param input 输入数据
 * @param input_shape 输入形状
 * @param output 输出数据 (由函数分配，需要调用者释放)
 * @param output_shape 输出形状
 * @return 0成功，非0失败
 */
int onnx_session_run_single(
    ONNXSession* session,
    const float* input,
    const TensorShape* input_shape,
    float** output,
    TensorShape* output_shape
);

/**
 * @brief 执行多输入多输出推理
 * @param session 会话句柄
 * @param inputs 输入张量数组
 * @param num_inputs 输入数量
 * @param outputs 输出张量数组 (由函数分配)
 * @param num_outputs 输出数量
 * @return 0成功，非0失败
 */
int onnx_session_run_multi(
    ONNXSession* session,
    const TensorData* inputs,
    size_t num_inputs,
    TensorData** outputs,
    size_t* num_outputs
);

// =============================================================================
// 工具函数
// =============================================================================

/**
 * @brief 释放张量数据
 * @param tensor 张量数据
 */
void tensor_data_free(TensorData* tensor);

/**
 * @brief 创建张量形状
 * @param dims 维度数组
 * @param num_dims 维度数量
 * @return 张量形状
 */
TensorShape tensor_shape_create(const int64_t* dims, size_t num_dims);

/**
 * @brief 释放张量形状
 * @param shape 张量形状
 */
void tensor_shape_free(TensorShape* shape);

/**
 * @brief 计算张量元素总数
 * @param shape 张量形状
 * @return 元素总数
 */
size_t tensor_shape_size(const TensorShape* shape);

#ifdef __cplusplus
}
#endif

#endif // ONNX_INFERENCE_H
