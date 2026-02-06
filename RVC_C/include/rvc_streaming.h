/**
 * @file rvc_streaming.h
 * @brief RVC 流式 (实时) 语音转换 API
 *
 * 基于 Python rtrvc.py 的实现，提供流式语音转换功能。
 * 支持分块处理音频，维护状态以保证音频连续性。
 */

#ifndef RVC_STREAMING_H
#define RVC_STREAMING_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// 前向声明
struct RVCContext;

// =============================================================================
// 常量定义
// =============================================================================

/** Pitch 缓存大小 (帧数) */
#define RVC_PITCH_CACHE_SIZE 1024

/** 默认音频上下文大小 (样本数, 300ms @ 16kHz) */
#define RVC_DEFAULT_AUDIO_CONTEXT_SIZE 4800

/** 默认处理块大小 (样本数, 1秒 @ 16kHz) */
#define RVC_DEFAULT_BLOCK_SIZE 16000

/** 默认交叉淡化样本数 (10ms @ 48kHz) */
#define RVC_DEFAULT_CROSSFADE_SAMPLES 480

/** ContentVec 特征维度 */
#define RVC_FEATURE_DIM 768

/** ContentVec hop size (样本数 @ 16kHz) */
#define RVC_CONTENTVEC_HOP_SIZE 320

/** 合成器 hop size (样本数 @ 16kHz) */
#define RVC_SYNTH_HOP_SIZE 160

// =============================================================================
// 类型定义
// =============================================================================

/** SOLA 重叠窗口大小 (样本数 @ 48kHz, ~10ms) */
#define RVC_SOLA_OVERLAP_SIZE 480

/** SOLA 搜索范围 (样本数 @ 48kHz) */
#define RVC_SOLA_SEARCH_RANGE 240

/** 流式处理状态 */
typedef struct RVCStreamState {
    // Pitch 缓存 (1024 帧)
    int64_t* cache_pitch;           /**< 量化音高缓存 [RVC_PITCH_CACHE_SIZE] */
    float* cache_pitchf;            /**< 连续音高缓存 [RVC_PITCH_CACHE_SIZE] */

    // 音频输入缓冲 (用于 F0 提取的额外上下文)
    float* audio_context;           /**< 前一块的尾部音频 */
    size_t audio_context_size;      /**< 音频上下文大小 (样本数) */
    size_t audio_context_capacity;  /**< 音频上下文容量 */

    // 特征缓冲 (用于重叠处理)
    float* feature_context;         /**< 前一块的尾部特征 */
    size_t feature_context_frames;  /**< 特征上下文帧数 */
    size_t feature_context_capacity;/**< 特征上下文容量 (帧数) */

    // SOLA 输出缓冲 (用于帧间平滑)
    float* output_tail;             /**< 前一块输出的尾部 (用于 SOLA) */
    size_t output_tail_size;        /**< 输出尾部大小 */
    size_t output_tail_capacity;    /**< 输出尾部容量 */

    // 状态标志
    int is_first_chunk;             /**< 是否第一块 (1=是, 0=否) */
    size_t total_samples_processed; /**< 已处理的总样本数 */
    size_t total_frames_processed;  /**< 已处理的总帧数 */

    // 内部工作缓冲区
    float* work_audio;              /**< 工作音频缓冲区 */
    size_t work_audio_capacity;     /**< 工作音频容量 */
    float* work_features;           /**< 工作特征缓冲区 */
    size_t work_features_capacity;  /**< 工作特征容量 */
} RVCStreamState;

/** 流式配置 */
typedef struct RVCStreamConfig {
    size_t block_size;              /**< 处理块大小 (样本数, 默认: 16000) */
    size_t audio_context_size;      /**< 音频上下文大小 (样本数, 默认: 4800) */
    size_t crossfade_samples;       /**< 交叉淡化样本数 (默认: 480) */
    int speaker_id;                 /**< 说话人 ID (默认: 0) */
} RVCStreamConfig;

/** 流式处理错误码 */
typedef enum {
    RVC_STREAM_SUCCESS = 0,
    RVC_STREAM_ERROR_INVALID_PARAM = -1,
    RVC_STREAM_ERROR_NOT_INITIALIZED = -2,
    RVC_STREAM_ERROR_MEMORY = -3,
    RVC_STREAM_ERROR_F0_EXTRACT = -4,
    RVC_STREAM_ERROR_FEATURE_EXTRACT = -5,
    RVC_STREAM_ERROR_SYNTHESIS = -6,
} RVCStreamError;

// =============================================================================
// 流式状态管理
// =============================================================================

/**
 * @brief 获取默认流式配置
 * @return 默认配置
 */
RVCStreamConfig rvc_stream_default_config(void);

/**
 * @brief 创建流式处理状态
 * @param config 流式配置 (可为 NULL 使用默认配置)
 * @return 状态句柄，失败返回 NULL
 */
RVCStreamState* rvc_stream_state_create(const RVCStreamConfig* config);

/**
 * @brief 销毁流式处理状态
 * @param state 状态句柄
 */
void rvc_stream_state_destroy(RVCStreamState* state);

/**
 * @brief 重置流式处理状态 (开始新的流)
 * @param state 状态句柄
 */
void rvc_stream_state_reset(RVCStreamState* state);

// =============================================================================
// 流式处理
// =============================================================================

/**
 * @brief 流式语音转换 (核心函数)
 *
 * 处理一块输入音频，返回转换后的音频。
 * 输入音频应为 16kHz 单声道，输出为 48kHz 单声道。
 *
 * @param ctx RVC 上下文
 * @param state 流式状态
 * @param input_chunk 输入音频块 (16kHz, float, -1.0~1.0)
 * @param input_samples 输入样本数
 * @param output_chunk 输出音频缓冲区 (48kHz)
 * @param output_capacity 输出缓冲区容量
 * @param output_samples 实际输出样本数 (输出参数)
 * @param config 流式配置
 * @return 错误码
 */
RVCStreamError rvc_stream_process(
    struct RVCContext* ctx,
    RVCStreamState* state,
    const float* input_chunk,
    size_t input_samples,
    float* output_chunk,
    size_t output_capacity,
    size_t* output_samples,
    const RVCStreamConfig* config
);

/**
 * @brief 刷新流式处理 (处理剩余数据)
 *
 * 在流结束时调用，处理缓冲区中剩余的数据。
 *
 * @param ctx RVC 上下文
 * @param state 流式状态
 * @param output_chunk 输出音频缓冲区
 * @param output_capacity 输出缓冲区容量
 * @param output_samples 实际输出样本数 (输出参数)
 * @param config 流式配置
 * @return 错误码
 */
RVCStreamError rvc_stream_flush(
    struct RVCContext* ctx,
    RVCStreamState* state,
    float* output_chunk,
    size_t output_capacity,
    size_t* output_samples,
    const RVCStreamConfig* config
);

// 前向声明 ONNX 类型
struct ONNXSession;
struct F0Extractor;

/**
 * @brief 独立的流式处理函数 (不依赖 RVCContext)
 *
 * 直接使用 ONNX sessions 和 F0 提取器进行流式处理。
 * 适用于需要更精细控制的场景。
 *
 * @param contentvec_session ContentVec/HuBERT ONNX 会话
 * @param synth_session 合成器 ONNX 会话
 * @param f0_extractor F0 提取器
 * @param state 流式状态
 * @param input_chunk 输入音频块 (16kHz)
 * @param input_samples 输入样本数
 * @param output_chunk 输出音频块 (48kHz)
 * @param output_capacity 输出缓冲区容量
 * @param output_samples 实际输出样本数 (输出参数)
 * @param speaker_id 说话人 ID
 * @param pitch_shift 音高偏移 (半音)
 * @return 错误码
 */
RVCStreamError rvc_stream_process_standalone(
    struct ONNXSession* contentvec_session,
    struct ONNXSession* synth_session,
    struct F0Extractor* f0_extractor,
    RVCStreamState* state,
    const float* input_chunk,
    size_t input_samples,
    float* output_chunk,
    size_t output_capacity,
    size_t* output_samples,
    int speaker_id,
    float pitch_shift
);

// =============================================================================
// 工具函数
// =============================================================================

/**
 * @brief 计算输出缓冲区所需大小
 *
 * 根据输入样本数计算输出缓冲区所需的最小大小。
 * 输出采样率为 48kHz，输入为 16kHz，因此输出约为输入的 3 倍。
 *
 * @param input_samples 输入样本数
 * @return 所需输出缓冲区大小 (样本数)
 */
size_t rvc_stream_calc_output_size(size_t input_samples);

/**
 * @brief 获取流式处理错误信息
 * @param error 错误码
 * @return 错误描述字符串
 */
const char* rvc_stream_error_string(RVCStreamError error);

/**
 * @brief 特征 2x 上采样 (线性插值)
 *
 * 将 ContentVec 特征从 hop_size=320 上采样到 hop_size=160。
 *
 * @param input 输入特征 [input_frames, feature_dim]
 * @param input_frames 输入帧数
 * @param output 输出特征 [input_frames*2, feature_dim]
 * @param feature_dim 特征维度 (通常为 768)
 */
void rvc_interpolate_features_2x(
    const float* input,
    size_t input_frames,
    float* output,
    size_t feature_dim
);

/**
 * @brief F0 转量化音高值
 *
 * 将 F0 (Hz) 转换为 RVC 使用的量化音高值 (1-255)。
 *
 * @param f0 F0 值 (Hz)
 * @param f0_min 最小 F0 (默认: 50)
 * @param f0_max 最大 F0 (默认: 1100)
 * @return 量化音高值 (1-255)
 */
int64_t rvc_f0_to_pitch(float f0, float f0_min, float f0_max);

/**
 * @brief 批量 F0 转量化音高
 *
 * @param f0 F0 数组 (Hz)
 * @param length 数组长度
 * @param pitch 输出量化音高数组
 * @param pitchf 输出连续音高数组
 * @param f0_min 最小 F0
 * @param f0_max 最大 F0
 */
void rvc_f0_to_pitch_batch(
    const double* f0,
    size_t length,
    int64_t* pitch,
    float* pitchf,
    float f0_min,
    float f0_max
);

#ifdef __cplusplus
}
#endif

#endif // RVC_STREAMING_H
