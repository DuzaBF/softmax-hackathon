#include "lib/rvv_softmax_f32.h"

#define SIZE 133

extern volatile uint32_t fromhost __attribute__((section(".fromhost")));
extern volatile uint32_t tohost __attribute__((section(".tohost")));

void check(const float *in, const float *out, const float *golden) {
    for (int i = 0; i < SIZE; ++i) {
        float diff = (golden[i] - out[i]);
        diff = diff > 0 ? diff : -diff;
    }
}

int main() {
    float in[SIZE] = {0.0f};
    float out[SIZE] = {0.0f};
    float golden[SIZE] = {0.0f};

    uint32_t size = SIZE;

    tohost = 0xDEADF00D;
    fromhost = 0xDEADF00D;

    rvv_softmax_f32(in, size, out);

    check(in, out, golden);
    while (1) {
    }
    return 0;
}