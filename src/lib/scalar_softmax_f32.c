#include "scalar_softmax_f32.h"

#include <assert.h>

static const float k1OverLn2 = 1.44269504089f;

typedef union {
    int32_t i;
    float flt;
} u_float_int_t;

float delta(const float in) {
    const float s1 = 0.30758037765820823f;
    const float s2 = -0.23141283591588344f;
    const float s3 = -7.6167541742324804e-2f;
    return s1 * in + s2 * in * in + s3 * in * in * in;
}

float fast_pow2(const float in) {
    assert(in <= 0.0f);
    const float yf = in - (int32_t)(in) + 1;
    u_float_int_t value;
    value.i = (int32_t)((1 << 23) * (in - delta(yf) + 127));
    return value.flt;
}

// Softmax Functions
int32_t scalar_softmax_f32(const float* in_vec, uint32_t size, float* out_vec) {
    float max = in_vec[0];
    float sum = 0;
    for (int i = 1; i < size; ++i) {
        if (in_vec[i] > max) {
            max = in_vec[i];
        }
    }
    for (int i = 0; i < size; ++i) {
        float value = (in_vec[i] - max) * k1OverLn2;
        out_vec[i] = fast_pow2(value);
        sum += out_vec[i];
    }
    for (int i = 0; i < size; ++i) {
        out_vec[i] = out_vec[i] / sum;
    }
    return 0;
}
