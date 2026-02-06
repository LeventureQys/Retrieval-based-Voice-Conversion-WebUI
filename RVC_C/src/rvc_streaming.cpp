/**
 * @file rvc_streaming.cpp
 * @brief RVC 流式 (实时) 语音转换实现
 *
 * 基于 Python rtrvc.py 的实现，提供流式语音转换功能。
 * 支持分块处理音频，维护状态以保证音频连续性。
 */

#include "rvc_streaming.h"
#include "rvc_onnx.h"
#include "onnx_inference.h"
#include "audio_processor.h"
#include "f0_extractor.h"
#include "utils.h"
#include "mt19937.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <algorithm>

// =============================================================================
// 常量定义
// =============================================================================

/** F0 转换参数 */
static const float F0_MIN_DEFAULT = 50.0f;
static const float F0_MAX_DEFAULT = 1100.0f;

// =============================================================================
// 工具函数
// =============================================================================

RVCStreamConfig rvc_stream_default_config(void) {
    RVCStreamConfig config;
    config.block_size = RVC_DEFAULT_BLOCK_SIZE;
    config.audio_context_size = RVC_DEFAULT_AUDIO_CONTEXT_SIZE;
    config.crossfade_samples = RVC_DEFAULT_CROSSFADE_SAMPLES;
    config.speaker_id = 0;
    return config;
}

size_t rvc_stream_calc_output_size(size_t input_samples) {
    // 输出为 48kHz，输入为 16kHz，比例为 3:1
    // 额外增加一些余量
    return (input_samples * 3) + 1024;
}

const char* rvc_stream_error_string(RVCStreamError error) {
    switch (error) {
        case RVC_STREAM_SUCCESS:
            return "Success";
        case RVC_STREAM_ERROR_INVALID_PARAM:
            return "Invalid parameter";
        case RVC_STREAM_ERROR_NOT_INITIALIZED:
            return "Not initialized";
        case RVC_STREAM_ERROR_MEMORY:
            return "Memory allocation failed";
        case RVC_STREAM_ERROR_F0_EXTRACT:
            return "F0 extraction failed";
        case RVC_STREAM_ERROR_FEATURE_EXTRACT:
            return "Feature extraction failed";
        case RVC_STREAM_ERROR_SYNTHESIS:
            return "Synthesis failed";
        default:
            return "Unknown error";
    }
}

// =============================================================================
// 流式状态管理
// =============================================================================

RVCStreamState* rvc_stream_state_create(const RVCStreamConfig* config) {
    RVCStreamConfig cfg;
    if (config) {
        cfg = *config;
    } else {
        cfg = rvc_stream_default_config();
    }

    RVCStreamState* state = (RVCStreamState*)calloc(1, sizeof(RVCStreamState));
    if (!state) {
        return NULL;
    }

    // 分配 Pitch 缓存
    state->cache_pitch = (int64_t*)calloc(RVC_PITCH_CACHE_SIZE, sizeof(int64_t));
    state->cache_pitchf = (float*)calloc(RVC_PITCH_CACHE_SIZE, sizeof(float));
    if (!state->cache_pitch || !state->cache_pitchf) {
        rvc_stream_state_destroy(state);
        return NULL;
    }

    // 分配音频上下文缓冲区
    state->audio_context_capacity = cfg.audio_context_size;
    state->audio_context = (float*)calloc(state->audio_context_capacity, sizeof(float));
    if (!state->audio_context) {
        rvc_stream_state_destroy(state);
        return NULL;
    }
    state->audio_context_size = 0;

    // 分配特征上下文缓冲区 (100 帧 * 768 维度)
    state->feature_context_capacity = 100;
    state->feature_context = (float*)calloc(
        state->feature_context_capacity * RVC_FEATURE_DIM, sizeof(float));
    if (!state->feature_context) {
        rvc_stream_state_destroy(state);
        return NULL;
    }
    state->feature_context_frames = 0;

    // 分配工作缓冲区
    // 工作音频缓冲区: 最大块大小 + 上下文
    state->work_audio_capacity = cfg.block_size + cfg.audio_context_size + 1024;
    state->work_audio = (float*)calloc(state->work_audio_capacity, sizeof(float));
    if (!state->work_audio) {
        rvc_stream_state_destroy(state);
        return NULL;
    }

    // 工作特征缓冲区: 足够容纳上采样后的特征
    state->work_features_capacity = 2048 * RVC_FEATURE_DIM;
    state->work_features = (float*)calloc(state->work_features_capacity, sizeof(float));
    if (!state->work_features) {
        rvc_stream_state_destroy(state);
        return NULL;
    }

    // 分配 SOLA 输出尾部缓冲区
    state->output_tail_capacity = RVC_SOLA_OVERLAP_SIZE + RVC_SOLA_SEARCH_RANGE;
    state->output_tail = (float*)calloc(state->output_tail_capacity, sizeof(float));
    if (!state->output_tail) {
        rvc_stream_state_destroy(state);
        return NULL;
    }
    state->output_tail_size = 0;

    // 初始化状态
    state->is_first_chunk = 1;
    state->total_samples_processed = 0;
    state->total_frames_processed = 0;

    return state;
}

void rvc_stream_state_destroy(RVCStreamState* state) {
    if (!state) return;

    if (state->cache_pitch) free(state->cache_pitch);
    if (state->cache_pitchf) free(state->cache_pitchf);
    if (state->audio_context) free(state->audio_context);
    if (state->feature_context) free(state->feature_context);
    if (state->work_audio) free(state->work_audio);
    if (state->work_features) free(state->work_features);
    if (state->output_tail) free(state->output_tail);

    free(state);
}

void rvc_stream_state_reset(RVCStreamState* state) {
    if (!state) return;

    // 重置 Pitch 缓存
    if (state->cache_pitch) {
        memset(state->cache_pitch, 0, RVC_PITCH_CACHE_SIZE * sizeof(int64_t));
    }
    if (state->cache_pitchf) {
        memset(state->cache_pitchf, 0, RVC_PITCH_CACHE_SIZE * sizeof(float));
    }

    // 重置上下文
    state->audio_context_size = 0;
    state->feature_context_frames = 0;

    // 重置 SOLA 输出缓冲
    state->output_tail_size = 0;

    // 重置状态
    state->is_first_chunk = 1;
    state->total_samples_processed = 0;
    state->total_frames_processed = 0;
}

// =============================================================================
// F0 转换函数
// =============================================================================

int64_t rvc_f0_to_pitch(float f0, float f0_min, float f0_max) {
    if (f0 <= 0.0f) {
        return 1;
    }

    // F0 转 Mel 刻度
    float f0_mel_min = 1127.0f * logf(1.0f + f0_min / 700.0f);
    float f0_mel_max = 1127.0f * logf(1.0f + f0_max / 700.0f);
    float f0_mel = 1127.0f * logf(1.0f + f0 / 700.0f);

    // 映射到 1-255
    float pitch = (f0_mel - f0_mel_min) * 254.0f / (f0_mel_max - f0_mel_min) + 1.0f;

    if (pitch < 1.0f) pitch = 1.0f;
    if (pitch > 255.0f) pitch = 255.0f;

    return (int64_t)roundf(pitch);
}

void rvc_f0_to_pitch_batch(
    const double* f0,
    size_t length,
    int64_t* pitch,
    float* pitchf,
    float f0_min,
    float f0_max
) {
    if (!f0 || !pitch || !pitchf || length == 0) return;

    float f0_mel_min = 1127.0f * logf(1.0f + f0_min / 700.0f);
    float f0_mel_max = 1127.0f * logf(1.0f + f0_max / 700.0f);

    for (size_t i = 0; i < length; i++) {
        float f0_val = (float)f0[i];

        if (f0_val <= 0.0f) {
            pitch[i] = 1;
            pitchf[i] = 0.0f;
        } else {
            // F0 转 Mel 刻度
            float f0_mel = 1127.0f * logf(1.0f + f0_val / 700.0f);

            // 映射到 1-255 (量化值)
            float p = (f0_mel - f0_mel_min) * 254.0f / (f0_mel_max - f0_mel_min) + 1.0f;
            if (p < 1.0f) p = 1.0f;
            if (p > 255.0f) p = 255.0f;
            pitch[i] = (int64_t)roundf(p);

            // 保存连续 F0 值
            pitchf[i] = f0_val;
        }
    }
}

// =============================================================================
// 特征上采样
// =============================================================================

void rvc_interpolate_features_2x(
    const float* input,
    size_t input_frames,
    float* output,
    size_t feature_dim
) {
    if (!input || !output || input_frames == 0 || feature_dim == 0) return;

    for (size_t i = 0; i < input_frames; i++) {
        const float* src_frame = input + i * feature_dim;
        float* dst_frame_0 = output + (i * 2) * feature_dim;
        float* dst_frame_1 = output + (i * 2 + 1) * feature_dim;

        // 复制原始帧
        memcpy(dst_frame_0, src_frame, feature_dim * sizeof(float));

        // 插值帧
        if (i < input_frames - 1) {
            const float* next_frame = input + (i + 1) * feature_dim;
            for (size_t d = 0; d < feature_dim; d++) {
                dst_frame_1[d] = (src_frame[d] + next_frame[d]) * 0.5f;
            }
        } else {
            // 最后一帧：复制
            memcpy(dst_frame_1, src_frame, feature_dim * sizeof(float));
        }
    }
}

// =============================================================================
// SOLA (Synchronous Overlap-Add) 算法
// =============================================================================

/**
 * @brief 计算两段音频的归一化互相关
 *
 * 用于找到最佳重叠位置，使帧间过渡更平滑
 *
 * @param ref 参考信号 (前一块的尾部)
 * @param ref_len 参考信号长度
 * @param search 搜索信号 (当前块的开头区域)
 * @param search_len 搜索信号长度
 * @return 最佳偏移量 (使相关性最大的位置)
 */
static size_t sola_find_best_offset(
    const float* ref,
    size_t ref_len,
    const float* search,
    size_t search_len
) {
    if (!ref || !search || ref_len == 0 || search_len <= ref_len) {
        return 0;
    }

    size_t search_range = search_len - ref_len;
    if (search_range > RVC_SOLA_SEARCH_RANGE) {
        search_range = RVC_SOLA_SEARCH_RANGE;
    }

    float best_corr = -1e10f;
    size_t best_offset = 0;

    for (size_t offset = 0; offset <= search_range; offset++) {
        // 计算归一化互相关
        float sum_xy = 0.0f;
        float sum_xx = 0.0f;
        float sum_yy = 0.0f;

        for (size_t i = 0; i < ref_len; i++) {
            float x = ref[i];
            float y = search[offset + i];
            sum_xy += x * y;
            sum_xx += x * x;
            sum_yy += y * y;
        }

        // 归一化
        float denom = sqrtf(sum_xx * sum_yy);
        float corr = (denom > 1e-8f) ? (sum_xy / denom) : 0.0f;

        if (corr > best_corr) {
            best_corr = corr;
            best_offset = offset;
        }
    }

    return best_offset;
}

/**
 * @brief 应用交叉淡化
 *
 * 将两段音频使用线性交叉淡化混合
 *
 * @param output 输出缓冲区
 * @param prev_tail 前一块的尾部
 * @param curr_start 当前块的开始 (已对齐)
 * @param fade_len 淡化长度
 */
static void sola_crossfade(
    float* output,
    const float* prev_tail,
    const float* curr_start,
    size_t fade_len
) {
    if (!output || !prev_tail || !curr_start || fade_len == 0) {
        return;
    }

    for (size_t i = 0; i < fade_len; i++) {
        float t = (float)i / (float)fade_len;  // 0 -> 1
        output[i] = prev_tail[i] * (1.0f - t) + curr_start[i] * t;
    }
}

// =============================================================================
// Pitch 缓存更新
// =============================================================================

/**
 * @brief 更新 Pitch 缓存
 *
 * 按照 Python 实现的逻辑：
 * - 缓存左移 shift 帧
 * - 新值从 pitch[3:-1] 插入到缓存末尾
 */
static void update_pitch_cache(
    RVCStreamState* state,
    const int64_t* new_pitch,
    const float* new_pitchf,
    size_t new_length,
    size_t shift
) {
    if (!state || !new_pitch || !new_pitchf || new_length <= 4) return;

    // 左移缓存
    if (shift > 0 && shift < RVC_PITCH_CACHE_SIZE) {
        memmove(state->cache_pitch,
                state->cache_pitch + shift,
                (RVC_PITCH_CACHE_SIZE - shift) * sizeof(int64_t));
        memmove(state->cache_pitchf,
                state->cache_pitchf + shift,
                (RVC_PITCH_CACHE_SIZE - shift) * sizeof(float));
    }

    // 插入新值 (跳过前 3 帧和最后 1 帧)
    // Python: self.cache_pitch[4 - pitch.shape[0]:] = pitch[3:-1]
    size_t insert_len = new_length - 4;  // pitch[3:-1] 的长度
    size_t insert_pos = RVC_PITCH_CACHE_SIZE - insert_len;

    if (insert_len > 0 && insert_pos < RVC_PITCH_CACHE_SIZE) {
        memcpy(state->cache_pitch + insert_pos,
               new_pitch + 3,
               insert_len * sizeof(int64_t));
        memcpy(state->cache_pitchf + insert_pos,
               new_pitchf + 3,
               insert_len * sizeof(float));
    }
}

// =============================================================================
// 流式处理核心 (独立函数版本)
// =============================================================================

/**
 * @brief 独立的流式处理函数 (不依赖 RVCContext)
 */
RVCStreamError rvc_stream_process_standalone(
    ONNXSession* contentvec_session,
    ONNXSession* synth_session,
    F0Extractor* f0_extractor,
    RVCStreamState* state,
    const float* input_chunk,
    size_t input_samples,
    float* output_chunk,
    size_t output_capacity,
    size_t* output_samples,
    int speaker_id,
    float pitch_shift
) {
    // 参数检查
    if (!contentvec_session || !synth_session || !f0_extractor ||
        !state || !input_chunk || !output_chunk || !output_samples) {
        return RVC_STREAM_ERROR_INVALID_PARAM;
    }

    if (input_samples == 0) {
        *output_samples = 0;
        return RVC_STREAM_SUCCESS;
    }

    int ret = 0;

    // =================================================================
    // 步骤 1: 拼接音频上下文
    // =================================================================
    size_t audio_with_context_size = state->audio_context_size + input_samples;

    if (audio_with_context_size > state->work_audio_capacity) {
        return RVC_STREAM_ERROR_MEMORY;
    }

    // 拼接: [audio_context, input_chunk]
    if (state->audio_context_size > 0) {
        memcpy(state->work_audio, state->audio_context,
               state->audio_context_size * sizeof(float));
    }
    memcpy(state->work_audio + state->audio_context_size, input_chunk,
           input_samples * sizeof(float));

    // =================================================================
    // 步骤 2: 提取 ContentVec 特征
    // =================================================================
    // 准备输入: [1, 1, audio_length]
    int64_t cv_input_dims[] = {1, 1, (int64_t)audio_with_context_size};
    TensorData cv_input;
    cv_input.data = state->work_audio;
    cv_input.size = audio_with_context_size;
    cv_input.shape = tensor_shape_create(cv_input_dims, 3);
    cv_input.dtype = TENSOR_TYPE_FLOAT32;

    TensorData* cv_outputs = NULL;
    size_t cv_num_outputs = 0;

    ret = onnx_session_run_multi(contentvec_session, &cv_input, 1, &cv_outputs, &cv_num_outputs);
    tensor_shape_free(&cv_input.shape);

    if (ret != 0 || cv_num_outputs == 0) {
        return RVC_STREAM_ERROR_FEATURE_EXTRACT;
    }

    // ContentVec 输出: [1, time_frames, 768]
    size_t cv_frames = cv_outputs[0].shape.dims[1];
    size_t cv_dim = cv_outputs[0].shape.dims[2];

    // =================================================================
    // 步骤 3: 提取 F0
    // =================================================================
    // F0 提取需要额外的上下文
    size_t f0_extractor_frame = input_samples + 800;
    size_t f0_start_pos = 0;
    if (audio_with_context_size > f0_extractor_frame) {
        f0_start_pos = audio_with_context_size - f0_extractor_frame;
    }
    size_t f0_audio_len = audio_with_context_size - f0_start_pos;

    F0Result f0_result;
    memset(&f0_result, 0, sizeof(F0Result));

    ret = f0_extract_float(f0_extractor,
                          state->work_audio + f0_start_pos,
                          f0_audio_len,
                          &f0_result);

    if (ret != 0) {
        for (size_t i = 0; i < cv_num_outputs; i++) {
            tensor_data_free(&cv_outputs[i]);
        }
        free(cv_outputs);
        return RVC_STREAM_ERROR_F0_EXTRACT;
    }

    // =================================================================
    // 步骤 4: 计算帧数和 skip_head/return_length
    // =================================================================
    size_t p_len = audio_with_context_size / RVC_SYNTH_HOP_SIZE;
    size_t synth_frames = cv_frames * 2;  // 2x 上采样后

    // 限制 p_len 不超过 synth_frames
    if (p_len > synth_frames) {
        p_len = synth_frames;
    }

    // skip_head: 跳过上下文部分的输出
    size_t skip_head = 0;
    if (!state->is_first_chunk && state->audio_context_size > 0) {
        skip_head = state->audio_context_size / RVC_SYNTH_HOP_SIZE;
    }

    // return_length: 实际需要返回的样本数
    size_t return_length = input_samples * 3;  // 16kHz -> 48kHz

    // =================================================================
    // 步骤 5: 更新 Pitch 缓存
    // =================================================================
    size_t shift = input_samples / RVC_SYNTH_HOP_SIZE;

    // 分配临时 pitch 缓冲区
    int64_t* temp_pitch = (int64_t*)malloc(f0_result.length * sizeof(int64_t));
    float* temp_pitchf = (float*)malloc(f0_result.length * sizeof(float));
    if (!temp_pitch || !temp_pitchf) {
        f0_result_free(&f0_result);
        for (size_t i = 0; i < cv_num_outputs; i++) {
            tensor_data_free(&cv_outputs[i]);
        }
        free(cv_outputs);
        if (temp_pitch) free(temp_pitch);
        if (temp_pitchf) free(temp_pitchf);
        return RVC_STREAM_ERROR_MEMORY;
    }

    // 应用音高偏移
    if (pitch_shift != 0.0f) {
        f0_shift_pitch(f0_result.f0, f0_result.length, pitch_shift);
    }

    // 转换 F0 到 pitch
    rvc_f0_to_pitch_batch(f0_result.f0, f0_result.length,
                          temp_pitch, temp_pitchf,
                          F0_MIN_DEFAULT, F0_MAX_DEFAULT);

    // 更新缓存
    update_pitch_cache(state, temp_pitch, temp_pitchf, f0_result.length, shift);

    free(temp_pitch);
    free(temp_pitchf);
    f0_result_free(&f0_result);

    // =================================================================
    // 步骤 6: 特征 2x 上采样
    // =================================================================
    float* cv_data = (float*)cv_outputs[0].data;

    // 检查工作缓冲区容量
    if (synth_frames * cv_dim > state->work_features_capacity) {
        for (size_t i = 0; i < cv_num_outputs; i++) {
            tensor_data_free(&cv_outputs[i]);
        }
        free(cv_outputs);
        return RVC_STREAM_ERROR_MEMORY;
    }

    rvc_interpolate_features_2x(cv_data, cv_frames, state->work_features, cv_dim);

    // =================================================================
    // 步骤 7: 准备合成器输入
    // =================================================================
    // 分配合成器输入
    float* phone = (float*)malloc(p_len * cv_dim * sizeof(float));
    int64_t* pitch = (int64_t*)malloc(p_len * sizeof(int64_t));
    float* pitchf = (float*)malloc(p_len * sizeof(float));
    float* rnd = (float*)malloc(192 * p_len * sizeof(float));

    if (!phone || !pitch || !pitchf || !rnd) {
        for (size_t i = 0; i < cv_num_outputs; i++) {
            tensor_data_free(&cv_outputs[i]);
        }
        free(cv_outputs);
        if (phone) free(phone);
        if (pitch) free(pitch);
        if (pitchf) free(pitchf);
        if (rnd) free(rnd);
        return RVC_STREAM_ERROR_MEMORY;
    }

    // 复制特征 (限制在 synth_frames 范围内)
    for (size_t t = 0; t < p_len; t++) {
        size_t src_t = t;
        if (src_t >= synth_frames) src_t = synth_frames - 1;
        memcpy(&phone[t * cv_dim], &state->work_features[src_t * cv_dim], cv_dim * sizeof(float));
    }

    // 从缓存中获取 pitch
    for (size_t t = 0; t < p_len; t++) {
        size_t cache_idx = RVC_PITCH_CACHE_SIZE - p_len + t;
        pitch[t] = state->cache_pitch[cache_idx];
        pitchf[t] = state->cache_pitchf[cache_idx];
    }

    // 生成随机噪声
    MT19937State mt_state;
    mt19937_seed(&mt_state, 42 + (unsigned int)state->total_samples_processed);
    for (size_t i = 0; i < 192 * p_len; i++) {
        rnd[i] = mt19937_randn_float(&mt_state);
    }

    // 准备合成器输入张量
    TensorData synth_inputs[6];
    int64_t phone_lengths_val = (int64_t)p_len;
    int64_t ds_val = (int64_t)speaker_id;

    // phone: [1, p_len, 768]
    int64_t phone_dims[] = {1, (int64_t)p_len, (int64_t)cv_dim};
    synth_inputs[0].data = phone;
    synth_inputs[0].size = p_len * cv_dim;
    synth_inputs[0].shape = tensor_shape_create(phone_dims, 3);
    synth_inputs[0].dtype = TENSOR_TYPE_FLOAT32;

    // phone_lengths: [1]
    int64_t phone_lengths_dims[] = {1};
    synth_inputs[1].data = &phone_lengths_val;
    synth_inputs[1].size = 1;
    synth_inputs[1].shape = tensor_shape_create(phone_lengths_dims, 1);
    synth_inputs[1].dtype = TENSOR_TYPE_INT64;

    // pitch: [1, p_len]
    int64_t pitch_dims[] = {1, (int64_t)p_len};
    synth_inputs[2].data = pitch;
    synth_inputs[2].size = p_len;
    synth_inputs[2].shape = tensor_shape_create(pitch_dims, 2);
    synth_inputs[2].dtype = TENSOR_TYPE_INT64;

    // pitchf: [1, p_len]
    int64_t pitchf_dims[] = {1, (int64_t)p_len};
    synth_inputs[3].data = pitchf;
    synth_inputs[3].size = p_len;
    synth_inputs[3].shape = tensor_shape_create(pitchf_dims, 2);
    synth_inputs[3].dtype = TENSOR_TYPE_FLOAT32;

    // ds: [1]
    int64_t ds_dims[] = {1};
    synth_inputs[4].data = &ds_val;
    synth_inputs[4].size = 1;
    synth_inputs[4].shape = tensor_shape_create(ds_dims, 1);
    synth_inputs[4].dtype = TENSOR_TYPE_INT64;

    // rnd: [1, 192, p_len]
    int64_t rnd_dims[] = {1, 192, (int64_t)p_len};
    synth_inputs[5].data = rnd;
    synth_inputs[5].size = 192 * p_len;
    synth_inputs[5].shape = tensor_shape_create(rnd_dims, 3);
    synth_inputs[5].dtype = TENSOR_TYPE_FLOAT32;

    // =================================================================
    // 步骤 8: 运行合成器
    // =================================================================
    TensorData* synth_outputs = NULL;
    size_t synth_num_outputs = 0;

    ret = onnx_session_run_multi(synth_session, synth_inputs, 6, &synth_outputs, &synth_num_outputs);

    // 释放输入形状
    for (int i = 0; i < 6; i++) {
        tensor_shape_free(&synth_inputs[i].shape);
    }
    free(phone);
    free(pitch);
    free(pitchf);
    free(rnd);

    for (size_t i = 0; i < cv_num_outputs; i++) {
        tensor_data_free(&cv_outputs[i]);
    }
    free(cv_outputs);

    if (ret != 0 || synth_num_outputs == 0) {
        return RVC_STREAM_ERROR_SYNTHESIS;
    }

    // =================================================================
    // 步骤 9: 提取输出并应用 SOLA 平滑
    // =================================================================
    float* synth_audio = (float*)synth_outputs[0].data;
    size_t synth_audio_size = synth_outputs[0].size;

    // 计算实际输出范围
    size_t skip_samples = skip_head * 480;  // 48kHz hop_size = 480
    size_t actual_output = return_length;

    if (skip_samples >= synth_audio_size) {
        skip_samples = 0;
        actual_output = synth_audio_size;
    } else if (skip_samples + actual_output > synth_audio_size) {
        actual_output = synth_audio_size - skip_samples;
    }

    if (actual_output > output_capacity) {
        actual_output = output_capacity;
    }

    // 获取当前块的起始位置
    float* curr_start = synth_audio + skip_samples;

    // SOLA 处理
    size_t output_offset = 0;

    if (state->is_first_chunk || state->output_tail_size == 0) {
        // 第一块: 直接复制输出
        memcpy(output_chunk, curr_start, actual_output * sizeof(float));
        *output_samples = actual_output;
    } else {
        // 后续块: 应用 SOLA 平滑
        size_t overlap_size = RVC_SOLA_OVERLAP_SIZE;
        if (overlap_size > state->output_tail_size) {
            overlap_size = state->output_tail_size;
        }
        if (overlap_size > actual_output) {
            overlap_size = actual_output;
        }

        // 在当前块开头搜索最佳重叠位置
        size_t search_len = overlap_size + RVC_SOLA_SEARCH_RANGE;
        if (search_len > actual_output) {
            search_len = actual_output;
        }

        // 查找最佳偏移
        size_t best_offset = sola_find_best_offset(
            state->output_tail + (state->output_tail_size - overlap_size),
            overlap_size,
            curr_start,
            search_len
        );

        // 应用交叉淡化
        // 先复制前一块的尾部 (非重叠部分)
        size_t pre_overlap = state->output_tail_size - overlap_size;
        if (pre_overlap > 0 && pre_overlap <= output_capacity) {
            // 通常不需要复制这部分，因为它已经在上一次输出了
            // 这里我们只处理重叠区域
        }

        // 交叉淡化重叠区域
        sola_crossfade(
            output_chunk,
            state->output_tail + (state->output_tail_size - overlap_size),
            curr_start + best_offset,
            overlap_size
        );
        output_offset = overlap_size;

        // 复制剩余部分
        size_t remaining = actual_output - best_offset - overlap_size;
        if (best_offset + overlap_size < actual_output) {
            remaining = actual_output - best_offset - overlap_size;
            if (output_offset + remaining > output_capacity) {
                remaining = output_capacity - output_offset;
            }
            memcpy(output_chunk + output_offset,
                   curr_start + best_offset + overlap_size,
                   remaining * sizeof(float));
            *output_samples = output_offset + remaining;
        } else {
            *output_samples = output_offset;
        }
    }

    // 保存当前输出的尾部用于下一次 SOLA
    size_t tail_to_save = state->output_tail_capacity;
    if (tail_to_save > actual_output) {
        tail_to_save = actual_output;
    }

    if (tail_to_save > 0) {
        // 保存当前输出的尾部
        memcpy(state->output_tail,
               synth_audio + skip_samples + actual_output - tail_to_save,
               tail_to_save * sizeof(float));
        state->output_tail_size = tail_to_save;
    }

    // 释放合成器输出
    for (size_t i = 0; i < synth_num_outputs; i++) {
        tensor_data_free(&synth_outputs[i]);
    }
    free(synth_outputs);

    // =================================================================
    // 步骤 10: 更新上下文
    // =================================================================
    // 保存当前输入的尾部作为下一块的上下文
    size_t context_to_save = state->audio_context_capacity;
    if (context_to_save > input_samples) {
        context_to_save = input_samples;
    }

    if (context_to_save > 0) {
        memcpy(state->audio_context,
               input_chunk + input_samples - context_to_save,
               context_to_save * sizeof(float));
        state->audio_context_size = context_to_save;
    }

    // 更新状态
    state->is_first_chunk = 0;
    state->total_samples_processed += input_samples;
    state->total_frames_processed += p_len;

    return RVC_STREAM_SUCCESS;
}

// =============================================================================
// 流式处理核心 (使用 RVCContext)
// =============================================================================

RVCStreamError rvc_stream_process(
    struct RVCContext* ctx,
    RVCStreamState* state,
    const float* input_chunk,
    size_t input_samples,
    float* output_chunk,
    size_t output_capacity,
    size_t* output_samples,
    const RVCStreamConfig* config
) {
    // 参数检查
    if (!ctx || !state || !input_chunk || !output_chunk || !output_samples) {
        return RVC_STREAM_ERROR_INVALID_PARAM;
    }

    // 这个函数需要从 RVCContext 中获取 ONNX sessions
    // 由于 RVCContext 是不透明的，这里返回错误
    // 实际使用应该调用 rvc_stream_process_standalone
    return RVC_STREAM_ERROR_NOT_INITIALIZED;
}

RVCStreamError rvc_stream_flush(
    struct RVCContext* ctx,
    RVCStreamState* state,
    float* output_chunk,
    size_t output_capacity,
    size_t* output_samples,
    const RVCStreamConfig* config
) {
    // 参数检查
    if (!ctx || !state || !output_chunk || !output_samples) {
        return RVC_STREAM_ERROR_INVALID_PARAM;
    }

    // 如果有剩余的音频上下文，可以选择处理它或返回静音
    *output_samples = 0;
    return RVC_STREAM_SUCCESS;
}
