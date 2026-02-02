/**
 * @file f0_extractor.cpp
 * @brief F0 (基频) 提取实现
 *
 * 使用 World Vocoder 库实现 Harvest 和 DIO 算法
 */

#include "f0_extractor.h"
#include "utils.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// World Vocoder headers
#include "world/harvest.h"
#include "world/dio.h"
#include "world/stonemask.h"

// =============================================================================
// 内部结构定义
// =============================================================================

struct F0Extractor {
    F0Method method;
    int sample_rate;
    F0Params params;
};

// =============================================================================
// F0 提取器
// =============================================================================

F0Params f0_default_params(void) {
    F0Params params;
    params.f0_floor = 71.0;
    params.f0_ceil = 800.0;
    params.frame_period = 5.0;
    params.speed = 1;
    return params;
}

F0Extractor* f0_extractor_create(F0Method method, int sample_rate) {
    F0Extractor* extractor = (F0Extractor*)malloc(sizeof(F0Extractor));
    if (!extractor) {
        return NULL;
    }

    extractor->method = method;
    extractor->sample_rate = sample_rate;
    extractor->params = f0_default_params();

    const char* method_name = (method == F0_METHOD_HARVEST) ? "Harvest" : "DIO";
    LOG_INFO("F0 extractor created: method=%s, sample_rate=%d", method_name, sample_rate);

    return extractor;
}

void f0_extractor_destroy(F0Extractor* extractor) {
    if (extractor) {
        free(extractor);
    }
}

void f0_extractor_set_params(F0Extractor* extractor, const F0Params* params) {
    if (extractor && params) {
        extractor->params = *params;
    }
}

// =============================================================================
// F0 提取
// =============================================================================

int f0_extract(
    F0Extractor* extractor,
    const double* audio,
    size_t audio_size,
    F0Result* result
) {
    if (!extractor || !audio || !result || audio_size == 0) {
        return -1;
    }

    int fs = extractor->sample_rate;
    double frame_period = extractor->params.frame_period;

    // 计算F0长度
    int f0_length = GetSamplesForDIO(fs, (int)audio_size, frame_period);
    if (f0_length <= 0) {
        LOG_ERROR("Invalid F0 length: %d", f0_length);
        return -2;
    }

    // 分配结果内存
    result->f0 = (double*)malloc(f0_length * sizeof(double));
    result->time_axis = (double*)malloc(f0_length * sizeof(double));

    if (!result->f0 || !result->time_axis) {
        if (result->f0) free(result->f0);
        if (result->time_axis) free(result->time_axis);
        return -3;
    }

    result->length = f0_length;
    result->sample_rate = fs;
    result->frame_period = frame_period;

    int64_t start_time = get_time_ms();

    if (extractor->method == F0_METHOD_HARVEST) {
        // Harvest 算法
        HarvestOption option;
        InitializeHarvestOption(&option);
        option.f0_floor = extractor->params.f0_floor;
        option.f0_ceil = extractor->params.f0_ceil;
        option.frame_period = frame_period;

        // Harvest 初步估计
        double* raw_f0 = (double*)malloc(f0_length * sizeof(double));
        if (!raw_f0) {
            free(result->f0);
            free(result->time_axis);
            return -3;
        }

        Harvest(audio, (int)audio_size, fs, &option, result->time_axis, raw_f0);

        // StoneMask 精细化 (与 Python 一致)
        StoneMask(audio, (int)audio_size, fs, result->time_axis, raw_f0, f0_length, result->f0);

        free(raw_f0);

    } else {
        // DIO 算法
        DioOption option;
        InitializeDioOption(&option);
        option.f0_floor = extractor->params.f0_floor;
        option.f0_ceil = extractor->params.f0_ceil;
        option.frame_period = frame_period;
        option.speed = extractor->params.speed;

        // DIO 初步估计
        double* raw_f0 = (double*)malloc(f0_length * sizeof(double));
        Dio(audio, (int)audio_size, fs, &option, result->time_axis, raw_f0);

        // StoneMask 精细化
        StoneMask(audio, (int)audio_size, fs, result->time_axis, raw_f0, f0_length, result->f0);

        free(raw_f0);
    }

    int64_t elapsed = get_time_ms() - start_time;
    LOG_INFO("F0 extraction completed: %d frames in %lld ms", f0_length, (long long)elapsed);

    return 0;
}

int f0_extract_float(
    F0Extractor* extractor,
    const float* audio,
    size_t audio_size,
    F0Result* result
) {
    if (!audio || audio_size == 0) {
        return -1;
    }

    // 转换为 double
    double* audio_double = (double*)malloc(audio_size * sizeof(double));
    if (!audio_double) {
        return -2;
    }

    for (size_t i = 0; i < audio_size; i++) {
        audio_double[i] = (double)audio[i];
    }

    int ret = f0_extract(extractor, audio_double, audio_size, result);

    free(audio_double);
    return ret;
}

void f0_result_free(F0Result* result) {
    if (result) {
        if (result->f0) {
            free(result->f0);
            result->f0 = NULL;
        }
        if (result->time_axis) {
            free(result->time_axis);
            result->time_axis = NULL;
        }
        result->length = 0;
    }
}

// =============================================================================
// F0 后处理
// =============================================================================

void f0_smooth(double* f0, size_t length, int window_size) {
    if (!f0 || length == 0 || window_size <= 1) {
        return;
    }

    double* temp = (double*)malloc(length * sizeof(double));
    if (!temp) return;

    int half_win = window_size / 2;

    for (size_t i = 0; i < length; i++) {
        double sum = 0.0;
        int count = 0;

        for (int j = -half_win; j <= half_win; j++) {
            int idx = (int)i + j;
            if (idx >= 0 && idx < (int)length && f0[idx] > 0) {
                sum += f0[idx];
                count++;
            }
        }

        temp[i] = (count > 0) ? sum / count : f0[i];
    }

    memcpy(f0, temp, length * sizeof(double));
    free(temp);
}

void f0_median_filter(double* f0, size_t length, int window_size) {
    if (!f0 || length == 0 || window_size <= 1) {
        return;
    }

    double* temp = (double*)malloc(length * sizeof(double));
    double* window = (double*)malloc(window_size * sizeof(double));
    if (!temp || !window) {
        if (temp) free(temp);
        if (window) free(window);
        return;
    }

    int half_win = window_size / 2;

    for (size_t i = 0; i < length; i++) {
        int count = 0;

        for (int j = -half_win; j <= half_win; j++) {
            int idx = (int)i + j;
            if (idx >= 0 && idx < (int)length) {
                window[count++] = f0[idx];
            }
        }

        // 简单排序找中值
        for (int a = 0; a < count - 1; a++) {
            for (int b = a + 1; b < count; b++) {
                if (window[a] > window[b]) {
                    double t = window[a];
                    window[a] = window[b];
                    window[b] = t;
                }
            }
        }

        temp[i] = window[count / 2];
    }

    memcpy(f0, temp, length * sizeof(double));
    free(temp);
    free(window);
}

void f0_interpolate(double* f0, size_t length) {
    if (!f0 || length == 0) {
        return;
    }

    // 完全按照 Python RVC 的 interpolate_f0 实现
    // 关键点：Python 不会跳过已处理的帧，每个无声帧都会被独立处理
    // 这意味着后面的处理可能会覆盖前面的结果

    // 创建原始数据的副本（Python 中的 data）
    double* data = (double*)malloc(length * sizeof(double));
    if (!data) return;
    memcpy(data, f0, length * sizeof(double));

    double last_value = 0.0;

    for (size_t i = 0; i < length; i++) {
        if (data[i] <= 0.0) {
            // 找到下一个有效值
            size_t j = i + 1;
            for (; j < length; j++) {
                if (data[j] > 0.0) {
                    break;
                }
            }

            if (j < length - 1) {
                // j 不是最后一个元素
                if (last_value > 0.0) {
                    // 有前一个有效值，线性插值
                    // Python: step = (data[j] - data[i-1]) / (j - i)
                    // Python: ip_data[k] = data[i-1] + step * (k - i + 1)
                    // 注意：使用 data[i-1]，不是 last_value
                    double prev_value = (i > 0) ? data[i - 1] : 0.0;
                    double step = (data[j] - prev_value) / (double)(j - i);
                    for (size_t k = i; k < j; k++) {
                        f0[k] = prev_value + step * (double)(k - i + 1);
                    }
                } else {
                    // 没有前一个有效值，用后面的值填充
                    for (size_t k = i; k < j; k++) {
                        f0[k] = data[j];
                    }
                }
            } else {
                // j 是最后一个元素或者没找到有效值，用 last_value 填充
                for (size_t k = i; k < length; k++) {
                    f0[k] = last_value;
                }
            }
        } else {
            f0[i] = data[i];
            last_value = data[i];
        }
    }

    free(data);
}

void f0_shift_pitch(double* f0, size_t length, double semitones) {
    if (!f0 || length == 0 || semitones == 0.0) {
        return;
    }

    // 半音转换系数: 2^(semitones/12)
    double ratio = pow(2.0, semitones / 12.0);

    for (size_t i = 0; i < length; i++) {
        if (f0[i] > 0) {
            f0[i] *= ratio;
        }
    }
}

void f0_resize(const double* f0, size_t src_length, size_t dst_length, double* output) {
    if (!f0 || !output || src_length == 0 || dst_length == 0) {
        return;
    }

    // 完全按照 Python 的 resize_f0 实现:
    // source[source < 0.001] = np.nan
    // target = np.interp(np.arange(0, len(source) * target_len, len(source)) / target_len,
    //                    np.arange(0, len(source)), source)
    // res = np.nan_to_num(target)

    // np.interp 的行为: 对于 NaN 值，结果也是 NaN
    // 然后 nan_to_num 将 NaN 转为 0

    for (size_t i = 0; i < dst_length; i++) {
        // 计算源数组中的位置
        // Python: np.arange(0, len(source) * target_len, len(source)) / target_len
        // 这等价于: i * src_length / dst_length
        double src_pos = (double)i * (double)src_length / (double)dst_length;
        size_t idx = (size_t)src_pos;
        double frac = src_pos - idx;

        if (idx + 1 < src_length) {
            double v0 = f0[idx];
            double v1 = f0[idx + 1];

            // 如果任一值无效 (<=0)，结果为 0 (模拟 NaN -> 0)
            if (v0 < 0.001 || v1 < 0.001) {
                output[i] = 0.0;
            } else {
                output[i] = v0 * (1.0 - frac) + v1 * frac;
            }
        } else if (idx < src_length) {
            output[i] = (f0[idx] < 0.001) ? 0.0 : f0[idx];
        } else {
            output[i] = (f0[src_length - 1] < 0.001) ? 0.0 : f0[src_length - 1];
        }
    }
}

// =============================================================================
// 工具函数
// =============================================================================

double freq_to_midi(double freq) {
    if (freq <= 0) return 0;
    return 69.0 + 12.0 * log2(freq / 440.0);
}

double midi_to_freq(double midi) {
    return 440.0 * pow(2.0, (midi - 69.0) / 12.0);
}

double f0_to_mel(double f0) {
    if (f0 <= 0) return 0;
    return 1127.0 * log(1.0 + f0 / 700.0);
}

double mel_to_f0(double mel) {
    return 700.0 * (exp(mel / 1127.0) - 1.0);
}
