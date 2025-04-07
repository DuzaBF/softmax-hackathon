#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef union {
    uint32_t hex;
    float flt;
} u_hex_fp32_t;

int main() {
    float x = 10.1;
    u_hex_fp32_t v;
    v.hex = (1 << 23) * (1.4426950409 * x + 126.94201519f);
    printf("%.02f\r\n", 1.4426950409 * x);
    printf("%.02f\r\n", 1.4426950409 * x + 126.94201519f);
    printf("%.02f\r\n", (1 << 23) * (1.4426950409 * x + 126.94201519f));
    printf("%d\r\n", (uint32_t)((1 << 23) * (1.4426950409 * x + 126.94201519f)));
    printf("%#010x\r\n", (uint32_t)((1 << 23) * (1.4426950409 * x + 126.94201519f)));
    printf("%#010x\r\n", v.hex);
    printf("x = %.02f v = %.02f exp = %.02f\r\n", x, v.flt, expf(x));
}