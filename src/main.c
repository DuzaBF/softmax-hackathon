#include <stdint.h>
#include <stdio.h>
#include <unistd.h>

#include "lib/scalar_softmax_f32.h"

#define SIZE 133

void read_floats(float *dst, const char *name) {
    FILE *fptr;
    fptr = fopen(name, "r");
    for (int i = 0; i < SIZE; ++i) {
        int res = fread((void *)dst, sizeof(float), 1, fptr);
        if (res == 0) {
            printf("error");
        }
        dst++;
    }
}

void check(const float *in, const float *out, const float *golden) {
    for (int i = 0; i < SIZE; ++i) {
        float diff = (golden[i] - out[i]);
        diff = diff > 0 ? diff : -diff;
        printf(
            "in[%3d]=%0.4f | out[%3d]=%0.4f | golden[%3d]=%0.4f | "
            "diff[%3d]=%0.10f\r\n",
            i, in[i], i, out[i], i, golden[i], i, diff);
    }
}

int main(int argc, char *argv[]) {
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
    scalar_softmax_f32(in, size, out);
    check(in, out, golden);
    return ret;
}