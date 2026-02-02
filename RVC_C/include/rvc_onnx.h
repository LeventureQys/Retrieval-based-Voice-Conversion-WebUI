/**
 * @file rvc_onnx.h
 * @brief RVC ONNX 主API头文件
 *
 * RVC (Retrieval-based Voice Conversion) C语言实现
 * 基于ONNX Runtime进行神经网络推理
 */

#ifndef RVC_ONNX_H
#define RVC_ONNX_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>

// =============================================================================
// 类型定义
// =============================================================================

/** RVC 上下文句柄 */
typedef struct RVCContext RVCContext;

/** 错误码 */
typedef enum {
    RVC_SUCCESS = 0,
    RVC_ERROR_INVALID_PARAM = -1,
    RVC_ERROR_MODEL_LOAD = -2,
    RVC_ERROR_INFERENCE = -3,
    RVC_ERROR_AUDIO_PROCESS = -4,
    RVC_ERROR_MEMORY = -5,
    RVC_ERROR_FILE_IO = -6,
    RVC_ERROR_NOT_INITIALIZED = -7,
} RVCError;

/** F0 提取方法 */
typedef enum {
    RVC_F0_HARVEST = 0,     /**< Harvest 算法 (World Vocoder) */
    RVC_F0_DIO = 1,         /**< DIO 算法 (World Vocoder) */
    RVC_F0_RMVPE = 2,       /**< RMVPE 神经网络 (需要ONNX模型) */
    RVC_F0_CREPE = 3,       /**< CREPE 神经网络 (需要ONNX模型) */
} RVCF0Method;

/** RVC 配置参数 */
typedef struct {
    const char* hubert_model_path;      /**< HuBERT ONNX模型路径 */
    const char* synthesizer_model_path; /**< 合成器ONNX模型路径 */
    const char* rmvpe_model_path;       /**< RMVPE ONNX模型路径 (可选) */

    int sample_rate;                    /**< 输入采样率 (默认: 16000) */
    int target_sample_rate;             /**< 输出采样率 (默认: 48000) */

    float pitch_shift;                  /**< 音高偏移 (半音, 默认: 0) */
    float index_rate;                   /**< 索引率 (0-1, 默认: 0) */

    RVCF0Method f0_method;              /**< F0提取方法 */
    int block_size;                     /**< 处理块大小 (默认: 2048) */
    int num_threads;                    /**< 推理线程数 (默认: 4) */
} RVCConfig;

// =============================================================================
// 初始化和销毁
// =============================================================================

/**
 * @brief 创建默认配置
 * @return 默认配置结构体
 */
RVCConfig rvc_default_config(void);

/**
 * @brief 创建RVC上下文
 * @param config 配置参数
 * @return RVC上下文句柄，失败返回NULL
 */
RVCContext* rvc_create(const RVCConfig* config);

/**
 * @brief 销毁RVC上下文
 * @param ctx RVC上下文句柄
 */
void rvc_destroy(RVCContext* ctx);

// =============================================================================
// 音频转换
// =============================================================================

/**
 * @brief 转换音频 (单次处理)
 * @param ctx RVC上下文
 * @param input 输入音频数据 (float, -1.0 ~ 1.0)
 * @param input_samples 输入样本数
 * @param output 输出音频缓冲区
 * @param output_samples 输出样本数 (输入时为缓冲区大小，输出时为实际样本数)
 * @return 错误码
 */
RVCError rvc_convert(
    RVCContext* ctx,
    const float* input,
    size_t input_samples,
    float* output,
    size_t* output_samples
);

/**
 * @brief 流式转换 (实时处理)
 * @param ctx RVC上下文
 * @param input_chunk 输入音频块
 * @param chunk_size 块大小
 * @param output_chunk 输出音频块
 * @param output_size 输出大小
 * @return 错误码
 */
RVCError rvc_stream_convert(
    RVCContext* ctx,
    const float* input_chunk,
    size_t chunk_size,
    float* output_chunk,
    size_t* output_size
);

// =============================================================================
// 参数设置
// =============================================================================

/**
 * @brief 设置音高偏移
 * @param ctx RVC上下文
 * @param semitones 半音数 (-12 ~ +12)
 * @return 错误码
 */
RVCError rvc_set_pitch_shift(RVCContext* ctx, float semitones);

/**
 * @brief 设置索引率
 * @param ctx RVC上下文
 * @param rate 索引率 (0.0 ~ 1.0)
 * @return 错误码
 */
RVCError rvc_set_index_rate(RVCContext* ctx, float rate);

/**
 * @brief 设置F0提取方法
 * @param ctx RVC上下文
 * @param method F0提取方法
 * @return 错误码
 */
RVCError rvc_set_f0_method(RVCContext* ctx, RVCF0Method method);

// =============================================================================
// 模型管理
// =============================================================================

/**
 * @brief 加载HuBERT模型
 * @param ctx RVC上下文
 * @param model_path 模型路径
 * @return 错误码
 */
RVCError rvc_load_hubert(RVCContext* ctx, const char* model_path);

/**
 * @brief 加载合成器模型
 * @param ctx RVC上下文
 * @param model_path 模型路径
 * @return 错误码
 */
RVCError rvc_load_synthesizer(RVCContext* ctx, const char* model_path);

// =============================================================================
// 工具函数
// =============================================================================

/**
 * @brief 获取错误信息
 * @param error 错误码
 * @return 错误描述字符串
 */
const char* rvc_error_string(RVCError error);

/**
 * @brief 获取版本信息
 * @return 版本字符串
 */
const char* rvc_version(void);

#ifdef __cplusplus
}
#endif

#endif // RVC_ONNX_H
