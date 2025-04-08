#include <stdint.h>
#include <stdio.h>

#include "lib/scalar_softmax_f32.h"

#define SIZE 132

int main() {
    float src[SIZE] = {0.0f};
    float out[SIZE] = {0.0f};
    float golden[SIZE] = {0.0f};
    (void) golden;
    uint32_t size = SIZE;
    scalar_softmax_f32(src, size, out);
}