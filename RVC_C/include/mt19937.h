/**
 * @file mt19937.h
 * @brief Mersenne Twister 随机数生成器 (与 numpy.random 兼容)
 */

#ifndef MT19937_H
#define MT19937_H

#include <stdint.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

// Mersenne Twister 状态
typedef struct {
    uint32_t mt[624];
    int mti;
    int has_gauss;
    double gauss_next;
} MT19937State;

// 初始化
static inline void mt19937_seed(MT19937State* state, uint32_t seed) {
    state->mt[0] = seed;
    for (int i = 1; i < 624; i++) {
        state->mt[i] = (1812433253UL * (state->mt[i-1] ^ (state->mt[i-1] >> 30)) + i);
    }
    state->mti = 624;
    state->has_gauss = 0;
    state->gauss_next = 0.0;
}

// 生成 32 位随机数
static inline uint32_t mt19937_uint32(MT19937State* state) {
    uint32_t y;
    static const uint32_t mag01[2] = {0x0UL, 0x9908b0dfUL};

    if (state->mti >= 624) {
        int kk;
        for (kk = 0; kk < 624 - 397; kk++) {
            y = (state->mt[kk] & 0x80000000UL) | (state->mt[kk+1] & 0x7fffffffUL);
            state->mt[kk] = state->mt[kk + 397] ^ (y >> 1) ^ mag01[y & 0x1UL];
        }
        for (; kk < 623; kk++) {
            y = (state->mt[kk] & 0x80000000UL) | (state->mt[kk+1] & 0x7fffffffUL);
            state->mt[kk] = state->mt[kk + (397 - 624)] ^ (y >> 1) ^ mag01[y & 0x1UL];
        }
        y = (state->mt[623] & 0x80000000UL) | (state->mt[0] & 0x7fffffffUL);
        state->mt[623] = state->mt[396] ^ (y >> 1) ^ mag01[y & 0x1UL];
        state->mti = 0;
    }

    y = state->mt[state->mti++];
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680UL;
    y ^= (y << 15) & 0xefc60000UL;
    y ^= (y >> 18);

    return y;
}

// 生成 [0, 1) 范围的双精度浮点数
static inline double mt19937_double(MT19937State* state) {
    uint32_t a = mt19937_uint32(state) >> 5;
    uint32_t b = mt19937_uint32(state) >> 6;
    return (a * 67108864.0 + b) * (1.0 / 9007199254740992.0);
}

// 生成标准正态分布随机数 (与 numpy.random.randn 兼容)
static inline double mt19937_randn(MT19937State* state) {
    if (state->has_gauss) {
        state->has_gauss = 0;
        return state->gauss_next;
    }

    double x1, x2, r2;
    do {
        x1 = 2.0 * mt19937_double(state) - 1.0;
        x2 = 2.0 * mt19937_double(state) - 1.0;
        r2 = x1 * x1 + x2 * x2;
    } while (r2 >= 1.0 || r2 == 0.0);

    double f = sqrt(-2.0 * log(r2) / r2);
    state->gauss_next = f * x1;
    state->has_gauss = 1;
    return f * x2;
}

// 生成单精度浮点数
static inline float mt19937_randn_float(MT19937State* state) {
    return (float)mt19937_randn(state);
}

#ifdef __cplusplus
}
#endif

#endif // MT19937_H
