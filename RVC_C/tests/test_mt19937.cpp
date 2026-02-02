/**
 * @file test_mt19937.cpp
 * @brief 测试 Mersenne Twister 随机数生成器
 */

#include <stdio.h>
#include "mt19937.h"

int main() {
    printf("Testing MT19937 random number generator...\n\n");

    MT19937State state;
    mt19937_seed(&state, 42);

    printf("mt19937_randn with seed=42:\n");
    for (int i = 0; i < 10; i++) {
        double val = mt19937_randn(&state);
        printf("  %d: %.10f\n", i, val);
    }

    printf("\nExpected (from numpy.random.randn):\n");
    printf("  0: 0.4967141530\n");
    printf("  1: -0.1382643012\n");
    printf("  2: 0.6476885381\n");
    printf("  3: 1.5230298564\n");
    printf("  4: -0.2341533747\n");
    printf("  5: -0.2341369569\n");
    printf("  6: 1.5792128155\n");
    printf("  7: 0.7674347292\n");
    printf("  8: -0.4694743859\n");
    printf("  9: 0.5425600436\n");

    return 0;
}
