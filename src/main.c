#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>

#include "lib/scalar_softmax_f32.h"
#include "lib/rvv_softmax_f32.h"
#ifndef SIZE
#define SIZE 2048 // Change golden image according to the size
#endif

void read_floats(float *dst, const char *name) {
    FILE *fptr;
    fptr = fopen(name, "r");
    int res = fread((void *)dst, sizeof(float), SIZE, fptr);
    if (res == 0) {
        printf("error %s res %d\r\n", name, res);
    }
#if 0
    for (int i = 0; i < SIZE; ++i) {
        u_float_int_t x;
        x.f32 = dst[i];
        printf("%s | dst[%d]: %#010x %.5f\r\n", name, i, x.i32, x.f32);
    }
#endif
}

void check(const float *in, const float *out, const float *golden) {
    float maxAbsDiff = 0.0f;
    float sumSquareErr = 0.0f;
    float squareGolden = 0.0f;
    for (int i = 0; i < SIZE; ++i) {
        float diff = (golden[i] - out[i]);
        diff = diff > 0 ? diff : -diff;
        maxAbsDiff = (diff > maxAbsDiff) ? diff : maxAbsDiff;
        sumSquareErr += diff * diff;
        squareGolden += golden[i] * golden[i];
#if 0
        printf(
            "in[%3d]=% 2.4f | out[%3d]=% 2.4f | golden[%3d]=% 2.4f | "
            "diff[%3d]=% 2.10f\r\n",
            i, in[i], i, out[i], i, golden[i], i, diff);
#endif
    }
    // calculate the SNR

    float snr = 10 * log10f(squareGolden / sumSquareErr);
    printf("------------------\n");
    printf("maxAbsDiff = %f\n", maxAbsDiff);
    printf("sumSquareErr = %f\n", sumSquareErr);
    printf("squareGolden = %f\n", squareGolden);
    printf("snr_result = %f\n", snr);

    float f_scores = sqrtf(snr) * 10.0f;

    printf("f_scores = %f\n", f_scores);
    if (snr >= 30.0f) {
        int final_scores =
            ((int)f_scores < 60.0f)
                ? 60.0f
                : ((int)f_scores > 100.0f ? 100.0f : (int)f_scores);
        printf("\n--> Test PASSED! Your scores = %d\n\n", final_scores);
    } else {
        printf("\n--> Test FAILED! Your scores = %d\n\n", (int)f_scores);
    }
}

int main(int argc, char *argv[]) {
    printf("Softmax size=%d\r\n", SIZE);
    float in[SIZE] = {0.0f};
    float out[SIZE] = {0.0f};
    float golden[SIZE] = {0.0f};

    int ret = 0;
    int opt;

    while ((opt = getopt(argc, argv, "f:g:")) != -1) {
        switch (opt) {
            case 'f':
                read_floats(in, optarg);
                break;
            case 'g':
                read_floats(golden, optarg);
                break;
            default:
                break;
        }
    }
    uint32_t size = SIZE;
#ifdef USE_RVV
    rvv_softmax_f32(in, size, out);
#else
    scalar_softmax_f32(in, size, out);
#endif
    check(in, out, golden);
    return ret;
}