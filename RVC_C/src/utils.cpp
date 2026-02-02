/**
 * @file utils.cpp
 * @brief 工具函数实现
 */

#include "utils.h"
#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#endif

// =============================================================================
// 全局变量
// =============================================================================

static LogLevel g_log_level = LOG_INFO;

// =============================================================================
// 内存管理
// =============================================================================

void* aligned_malloc(size_t size, size_t alignment) {
#ifdef _WIN32
    return _aligned_malloc(size, alignment);
#else
    void* ptr = nullptr;
    if (posix_memalign(&ptr, alignment, size) != 0) {
        return nullptr;
    }
    return ptr;
#endif
}

void aligned_free(void* ptr) {
#ifdef _WIN32
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

// =============================================================================
// 数学工具
// =============================================================================

size_t next_power_of_2(size_t n) {
    if (n == 0) return 1;
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
#if SIZE_MAX > 0xFFFFFFFF
    n |= n >> 32;
#endif
    return n + 1;
}

int is_power_of_2(size_t n) {
    return n > 0 && (n & (n - 1)) == 0;
}

float lerp(float a, float b, float t) {
    return a + t * (b - a);
}

float clamp(float value, float min_val, float max_val) {
    if (value < min_val) return min_val;
    if (value > max_val) return max_val;
    return value;
}

// =============================================================================
// 数组操作
// =============================================================================

float array_sum(const float* arr, size_t size) {
    float sum = 0.0f;
    for (size_t i = 0; i < size; i++) {
        sum += arr[i];
    }
    return sum;
}

float array_mean(const float* arr, size_t size) {
    if (size == 0) return 0.0f;
    return array_sum(arr, size) / (float)size;
}

float array_max(const float* arr, size_t size) {
    if (size == 0) return 0.0f;
    float max_val = arr[0];
    for (size_t i = 1; i < size; i++) {
        if (arr[i] > max_val) {
            max_val = arr[i];
        }
    }
    return max_val;
}

float array_min(const float* arr, size_t size) {
    if (size == 0) return 0.0f;
    float min_val = arr[0];
    for (size_t i = 1; i < size; i++) {
        if (arr[i] < min_val) {
            min_val = arr[i];
        }
    }
    return min_val;
}

float array_std(const float* arr, size_t size) {
    if (size == 0) return 0.0f;
    float mean = array_mean(arr, size);
    float sum_sq = 0.0f;
    for (size_t i = 0; i < size; i++) {
        float diff = arr[i] - mean;
        sum_sq += diff * diff;
    }
    return sqrtf(sum_sq / (float)size);
}

void array_normalize(float* arr, size_t size) {
    if (size == 0) return;
    float max_abs = 0.0f;
    for (size_t i = 0; i < size; i++) {
        float abs_val = fabsf(arr[i]);
        if (abs_val > max_abs) {
            max_abs = abs_val;
        }
    }
    if (max_abs > 1e-8f) {
        for (size_t i = 0; i < size; i++) {
            arr[i] /= max_abs;
        }
    }
}

void array_scale(float* arr, size_t size, float scale) {
    for (size_t i = 0; i < size; i++) {
        arr[i] *= scale;
    }
}

// =============================================================================
// 时间测量
// =============================================================================

int64_t get_time_ms(void) {
#ifdef _WIN32
    LARGE_INTEGER freq, counter;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&counter);
    return (int64_t)(counter.QuadPart * 1000 / freq.QuadPart);
#else
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (int64_t)tv.tv_sec * 1000 + tv.tv_usec / 1000;
#endif
}

int64_t get_time_us(void) {
#ifdef _WIN32
    LARGE_INTEGER freq, counter;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&counter);
    return (int64_t)(counter.QuadPart * 1000000 / freq.QuadPart);
#else
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (int64_t)tv.tv_sec * 1000000 + tv.tv_usec;
#endif
}

// =============================================================================
// 日志
// =============================================================================

void log_set_level(LogLevel level) {
    g_log_level = level;
}

void log_message(LogLevel level, const char* fmt, ...) {
    if (level < g_log_level) {
        return;
    }

    const char* level_str;
    switch (level) {
        case LOG_DEBUG:   level_str = "DEBUG"; break;
        case LOG_INFO:    level_str = "INFO"; break;
        case LOG_WARNING: level_str = "WARNING"; break;
        case LOG_ERROR:   level_str = "ERROR"; break;
        default:          level_str = "UNKNOWN"; break;
    }

    // 获取时间
    time_t now = time(NULL);
    struct tm* tm_info = localtime(&now);
    char time_buf[32];
    strftime(time_buf, sizeof(time_buf), "%H:%M:%S", tm_info);

    // 输出日志
    fprintf(stderr, "[%s] [%s] ", time_buf, level_str);

    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);

    fprintf(stderr, "\n");
    fflush(stderr);
}
