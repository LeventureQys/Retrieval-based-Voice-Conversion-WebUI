/**
 * @file rvc_onnx.cpp
 * @brief RVC ONNX 主实现
 */

#include "rvc_onnx.h"
#include "onnx_inference.h"
#include "audio_processor.h"
#include "f0_extractor.h"
#include "stft.h"
#include "utils.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>

// =============================================================================
// 内部结构定义
// =============================================================================

struct RVCContext {
    RVCConfig config;

    // ONNX 引擎和会话
    ONNXEngine* engine;
    ONNXSession* hubert_session;
    ONNXSession* synthesizer_session;
    ONNXSession* rmvpe_session;

    // 处理模块
    AudioProcessor* audio_processor;
    F0Extractor* f0_extractor;
    STFTProcessor* stft_processor;

    // 内部缓冲区
    float* audio_buffer;
    size_t audio_buffer_size;
    float* feature_buffer;
    size_t feature_buffer_size;

    // 状态
    int initialized;
};

// =============================================================================
// 初始化和销毁
// =============================================================================

RVCConfig rvc_default_config(void) {
    RVCConfig config;
    memset(&config, 0, sizeof(RVCConfig));

    config.hubert_model_path = NULL;
    config.synthesizer_model_path = NULL;
    config.rmvpe_model_path = NULL;

    config.sample_rate = 16000;
    config.target_sample_rate = 48000;
    config.pitch_shift = 0.0f;
    config.index_rate = 0.0f;
    config.f0_method = RVC_F0_HARVEST;
    config.block_size = 2048;
    config.num_threads = 4;

    return config;
}

RVCContext* rvc_create(const RVCConfig* config) {
    if (!config) {
        LOG_ERROR("Invalid config");
        return NULL;
    }

    RVCContext* ctx = (RVCContext*)calloc(1, sizeof(RVCContext));
    if (!ctx) {
        LOG_ERROR("Failed to allocate RVC context");
        return NULL;
    }

    // 复制配置
    ctx->config = *config;
    ctx->initialized = 0;

    // 创建 ONNX 引擎
    ctx->engine = onnx_engine_create(config->num_threads);
    if (!ctx->engine) {
        LOG_ERROR("Failed to create ONNX engine");
        rvc_destroy(ctx);
        return NULL;
    }

    // 加载 HuBERT 模型
    if (config->hubert_model_path) {
        ctx->hubert_session = onnx_session_create(ctx->engine, config->hubert_model_path);
        if (!ctx->hubert_session) {
            LOG_ERROR("Failed to load HuBERT model: %s", config->hubert_model_path);
            rvc_destroy(ctx);
            return NULL;
        }
    }

    // 加载合成器模型
    if (config->synthesizer_model_path) {
        ctx->synthesizer_session = onnx_session_create(ctx->engine, config->synthesizer_model_path);
        if (!ctx->synthesizer_session) {
            LOG_ERROR("Failed to load synthesizer model: %s", config->synthesizer_model_path);
            rvc_destroy(ctx);
            return NULL;
        }
    }

    // 加载 RMVPE 模型 (可选)
    if (config->rmvpe_model_path && config->f0_method == RVC_F0_RMVPE) {
        ctx->rmvpe_session = onnx_session_create(ctx->engine, config->rmvpe_model_path);
        if (!ctx->rmvpe_session) {
            LOG_WARNING("Failed to load RMVPE model, falling back to Harvest");
            ctx->config.f0_method = RVC_F0_HARVEST;
        }
    }

    // 创建音频处理器
    ctx->audio_processor = audio_processor_create(config->sample_rate);
    if (!ctx->audio_processor) {
        LOG_ERROR("Failed to create audio processor");
        rvc_destroy(ctx);
        return NULL;
    }

    // 创建 F0 提取器
    F0Method f0_method = (config->f0_method == RVC_F0_DIO) ? F0_METHOD_DIO : F0_METHOD_HARVEST;
    ctx->f0_extractor = f0_extractor_create(f0_method, config->sample_rate);
    if (!ctx->f0_extractor) {
        LOG_ERROR("Failed to create F0 extractor");
        rvc_destroy(ctx);
        return NULL;
    }

    // 创建 STFT 处理器
    ctx->stft_processor = stft_processor_create(2048, 512, WINDOW_HANN);
    if (!ctx->stft_processor) {
        LOG_ERROR("Failed to create STFT processor");
        rvc_destroy(ctx);
        return NULL;
    }

    // 分配内部缓冲区
    ctx->audio_buffer_size = config->sample_rate * 30; // 30秒缓冲
    ctx->audio_buffer = (float*)malloc(ctx->audio_buffer_size * sizeof(float));

    ctx->feature_buffer_size = 1024 * 768; // 特征缓冲
    ctx->feature_buffer = (float*)malloc(ctx->feature_buffer_size * sizeof(float));

    if (!ctx->audio_buffer || !ctx->feature_buffer) {
        LOG_ERROR("Failed to allocate buffers");
        rvc_destroy(ctx);
        return NULL;
    }

    ctx->initialized = 1;
    LOG_INFO("RVC context created successfully");

    return ctx;
}

void rvc_destroy(RVCContext* ctx) {
    if (!ctx) return;

    // 销毁 ONNX 会话
    if (ctx->hubert_session) onnx_session_destroy(ctx->hubert_session);
    if (ctx->synthesizer_session) onnx_session_destroy(ctx->synthesizer_session);
    if (ctx->rmvpe_session) onnx_session_destroy(ctx->rmvpe_session);

    // 销毁 ONNX 引擎
    if (ctx->engine) onnx_engine_destroy(ctx->engine);

    // 销毁处理模块
    if (ctx->audio_processor) audio_processor_destroy(ctx->audio_processor);
    if (ctx->f0_extractor) f0_extractor_destroy(ctx->f0_extractor);
    if (ctx->stft_processor) stft_processor_destroy(ctx->stft_processor);

    // 释放缓冲区
    if (ctx->audio_buffer) free(ctx->audio_buffer);
    if (ctx->feature_buffer) free(ctx->feature_buffer);

    free(ctx);
    LOG_INFO("RVC context destroyed");
}

// =============================================================================
// 音频转换
// =============================================================================

RVCError rvc_convert(
    RVCContext* ctx,
    const float* input,
    size_t input_samples,
    float* output,
    size_t* output_samples
) {
    if (!ctx || !ctx->initialized) {
        return RVC_ERROR_NOT_INITIALIZED;
    }

    if (!input || !output || !output_samples || input_samples == 0) {
        return RVC_ERROR_INVALID_PARAM;
    }

    int64_t start_time = get_time_ms();

    // 1. 音频预处理
    float* processed_audio = (float*)malloc(input_samples * sizeof(float));
    if (!processed_audio) {
        return RVC_ERROR_MEMORY;
    }

    size_t processed_size;
    int ret = audio_preprocess(ctx->audio_processor, input, input_samples,
                               processed_audio, &processed_size);
    if (ret != 0) {
        free(processed_audio);
        return RVC_ERROR_AUDIO_PROCESS;
    }

    // 2. 提取 F0
    F0Result f0_result;
    memset(&f0_result, 0, sizeof(F0Result));

    ret = f0_extract_float(ctx->f0_extractor, processed_audio, processed_size, &f0_result);
    if (ret != 0) {
        free(processed_audio);
        return RVC_ERROR_AUDIO_PROCESS;
    }

    // 应用音高偏移
    if (ctx->config.pitch_shift != 0.0f) {
        f0_shift_pitch(f0_result.f0, f0_result.length, ctx->config.pitch_shift);
    }

    // 3. 提取 HuBERT 特征 (如果模型已加载)
    float* hubert_features = NULL;
    size_t hubert_feature_size = 0;

    if (ctx->hubert_session) {
        // 准备输入形状 [1, audio_length]
        int64_t input_dims[] = {1, (int64_t)processed_size};
        TensorShape input_shape = tensor_shape_create(input_dims, 2);

        TensorShape output_shape;
        memset(&output_shape, 0, sizeof(TensorShape));

        ret = onnx_session_run_single(ctx->hubert_session, processed_audio,
                                      &input_shape, &hubert_features, &output_shape);

        tensor_shape_free(&input_shape);

        if (ret != 0) {
            LOG_ERROR("HuBERT inference failed");
            f0_result_free(&f0_result);
            free(processed_audio);
            return RVC_ERROR_INFERENCE;
        }

        hubert_feature_size = tensor_shape_size(&output_shape);
        tensor_shape_free(&output_shape);
    }

    // 4. 合成音频 (如果合成器模型已加载)
    if (ctx->synthesizer_session && hubert_features) {
        // TODO: 实现完整的合成器推理
        // 这需要根据具体的 RVC 模型结构来实现
        // 包括特征处理、F0 编码、合成器推理等

        LOG_INFO("Synthesizer inference - TODO: implement full pipeline");
    }

    // 5. 目前简单地复制输入到输出 (占位实现)
    // 实际实现需要完成上述合成步骤
    size_t copy_size = input_samples;
    if (copy_size > *output_samples) {
        copy_size = *output_samples;
    }
    memcpy(output, input, copy_size * sizeof(float));
    *output_samples = copy_size;

    // 清理
    if (hubert_features) free(hubert_features);
    f0_result_free(&f0_result);
    free(processed_audio);

    int64_t elapsed = get_time_ms() - start_time;
    LOG_INFO("Conversion completed in %lld ms", (long long)elapsed);

    return RVC_SUCCESS;
}

RVCError rvc_stream_convert(
    RVCContext* ctx,
    const float* input_chunk,
    size_t chunk_size,
    float* output_chunk,
    size_t* output_size
) {
    // 流式处理的简化实现
    // 实际实现需要维护状态和缓冲区
    return rvc_convert(ctx, input_chunk, chunk_size, output_chunk, output_size);
}

// =============================================================================
// 参数设置
// =============================================================================

RVCError rvc_set_pitch_shift(RVCContext* ctx, float semitones) {
    if (!ctx) return RVC_ERROR_INVALID_PARAM;

    ctx->config.pitch_shift = semitones;
    LOG_INFO("Pitch shift set to %.1f semitones", semitones);
    return RVC_SUCCESS;
}

RVCError rvc_set_index_rate(RVCContext* ctx, float rate) {
    if (!ctx) return RVC_ERROR_INVALID_PARAM;

    if (rate < 0.0f) rate = 0.0f;
    if (rate > 1.0f) rate = 1.0f;

    ctx->config.index_rate = rate;
    LOG_INFO("Index rate set to %.2f", rate);
    return RVC_SUCCESS;
}

RVCError rvc_set_f0_method(RVCContext* ctx, RVCF0Method method) {
    if (!ctx) return RVC_ERROR_INVALID_PARAM;

    ctx->config.f0_method = method;

    // 重新创建 F0 提取器
    if (ctx->f0_extractor) {
        f0_extractor_destroy(ctx->f0_extractor);
    }

    F0Method f0_method = (method == RVC_F0_DIO) ? F0_METHOD_DIO : F0_METHOD_HARVEST;
    ctx->f0_extractor = f0_extractor_create(f0_method, ctx->config.sample_rate);

    if (!ctx->f0_extractor) {
        return RVC_ERROR_MEMORY;
    }

    return RVC_SUCCESS;
}

// =============================================================================
// 模型管理
// =============================================================================

RVCError rvc_load_hubert(RVCContext* ctx, const char* model_path) {
    if (!ctx || !model_path) return RVC_ERROR_INVALID_PARAM;

    if (ctx->hubert_session) {
        onnx_session_destroy(ctx->hubert_session);
    }

    ctx->hubert_session = onnx_session_create(ctx->engine, model_path);
    if (!ctx->hubert_session) {
        return RVC_ERROR_MODEL_LOAD;
    }

    return RVC_SUCCESS;
}

RVCError rvc_load_synthesizer(RVCContext* ctx, const char* model_path) {
    if (!ctx || !model_path) return RVC_ERROR_INVALID_PARAM;

    if (ctx->synthesizer_session) {
        onnx_session_destroy(ctx->synthesizer_session);
    }

    ctx->synthesizer_session = onnx_session_create(ctx->engine, model_path);
    if (!ctx->synthesizer_session) {
        return RVC_ERROR_MODEL_LOAD;
    }

    return RVC_SUCCESS;
}

// =============================================================================
// 工具函数
// =============================================================================

const char* rvc_error_string(RVCError error) {
    switch (error) {
        case RVC_SUCCESS:              return "Success";
        case RVC_ERROR_INVALID_PARAM:  return "Invalid parameter";
        case RVC_ERROR_MODEL_LOAD:     return "Model load failed";
        case RVC_ERROR_INFERENCE:      return "Inference failed";
        case RVC_ERROR_AUDIO_PROCESS:  return "Audio processing failed";
        case RVC_ERROR_MEMORY:         return "Memory allocation failed";
        case RVC_ERROR_FILE_IO:        return "File I/O error";
        case RVC_ERROR_NOT_INITIALIZED: return "Not initialized";
        default:                       return "Unknown error";
    }
}

const char* rvc_version(void) {
    return "RVC_ONNX v1.0.0";
}
