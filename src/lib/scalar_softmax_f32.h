#pragma once

#include <stdint.h>

typedef union {
    int32_t i32;
    float f32;
} u_float_int_t;

int32_t scalar_softmax_f32(const float* in_vec, uint32_t size, float* out_vec);
