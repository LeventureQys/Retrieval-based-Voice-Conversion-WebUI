/**
 * @file utils.h
 * @brief 工具函数
 */

#ifndef UTILS_H
#define UTILS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// 内存管理
// =============================================================================

/**
 * @brief 分配对齐内存
 * @param size 大小
 * @param alignment 对齐字节数
 * @return 内存指针
 */
void* aligned_malloc(size_t size, size_t alignment);

/**
 * @brief 释放对齐内存
 * @param ptr 内存指针
 */
void aligned_free(void* ptr);

// =============================================================================
// 数学工具
// =============================================================================

/**
 * @brief 计算下一个2的幂
 * @param n 输入值
 * @return 大于等于n的最小2的幂
 */
size_t next_power_of_2(size_t n);

/**
 * @brief 是否为2的幂
 * @param n 输入值
 * @return 1是，0否
 */
int is_power_of_2(size_t n);

/**
 * @brief 线性插值
 * @param a 起始值
 * @param b 结束值
 * @param t 插值因子 (0-1)
 * @return 插值结果
 */
float lerp(float a, float b, float t);

/**
 * @brief 限制值在范围内
 * @param value 输入值
 * @param min_val 最小值
 * @param max_val 最大值
 * @return 限制后的值
 */
float clamp(float value, float min_val, float max_val);

// =============================================================================
// 数组操作
// =============================================================================

/**
 * @brief 数组求和
 * @param arr 数组
 * @param size 大小
 * @return 和
 */
float array_sum(const float* arr, size_t size);

/**
 * @brief 数组均值
 * @param arr 数组
 * @param size 大小
 * @return 均值
 */
float array_mean(const float* arr, size_t size);

/**
 * @brief 数组最大值
 * @param arr 数组
 * @param size 大小
 * @return 最大值
 */
float array_max(const float* arr, size_t size);

/**
 * @brief 数组最小值
 * @param arr 数组
 * @param size 大小
 * @return 最小值
 */
float array_min(const float* arr, size_t size);

/**
 * @brief 数组标准差
 * @param arr 数组
 * @param size 大小
 * @return 标准差
 */
float array_std(const float* arr, size_t size);

/**
 * @brief 数组归一化 (原地)
 * @param arr 数组
 * @param size 大小
 */
void array_normalize(float* arr, size_t size);

/**
 * @brief 数组缩放 (原地)
 * @param arr 数组
 * @param size 大小
 * @param scale 缩放因子
 */
void array_scale(float* arr, size_t size, float scale);

// =============================================================================
// 时间测量
// =============================================================================

/**
 * @brief 获取当前时间 (毫秒)
 * @return 时间戳
 */
int64_t get_time_ms(void);

/**
 * @brief 获取当前时间 (微秒)
 * @return 时间戳
 */
int64_t get_time_us(void);

// =============================================================================
// 日志
// =============================================================================

/** 日志级别 */
typedef enum {
    LOG_DEBUG = 0,
    LOG_INFO = 1,
    LOG_WARNING = 2,
    LOG_ERROR = 3,
} LogLevel;

/**
 * @brief 设置日志级别
 * @param level 日志级别
 */
void log_set_level(LogLevel level);

/**
 * @brief 输出日志
 * @param level 日志级别
 * @param fmt 格式字符串
 * @param ... 参数
 */
void log_message(LogLevel level, const char* fmt, ...);

#define LOG_DEBUG(fmt, ...) log_message(LOG_DEBUG, fmt, ##__VA_ARGS__)
#define LOG_INFO(fmt, ...) log_message(LOG_INFO, fmt, ##__VA_ARGS__)
#define LOG_WARNING(fmt, ...) log_message(LOG_WARNING, fmt, ##__VA_ARGS__)
#define LOG_ERROR(fmt, ...) log_message(LOG_ERROR, fmt, ##__VA_ARGS__)

#ifdef __cplusplus
}
#endif

#endif // UTILS_H
