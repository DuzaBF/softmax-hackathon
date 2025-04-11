#include "scalar_softmax_f32.h"

#include <stdio.h>

// #define CORR_POL_0
// #define CORR_POL_3
#define CORR_POL_5

static const float k1OverLn2 = 1.44269504089f;

#ifdef CORR_POL_5
static float delta(const float in) {
    const float s1 = 3.06852819440055e-1f;
    const float s2 = -2.40226506959101e-1f;
    const float s3 = -5.57129652016652e-2f;
    const float s4 = -9.01146535969578e-3f;
    const float s5 = -1.90188191959304e-3f;
    const float p1 = in;
    const float p2 = p1 * in;
    const float p3 = p2 * in;
    const float p4 = p3 * in;
    const float p5 = p4 * in;
    return s1 * p1 + s2 * p2 + s3 * p3 + s4 * p4 + s5 * p5;
}
#elif defined(CORR_POL_3)
static float delta(const float in) {
    const float s1 = 0.30758037765820823f;
    const float s2 = -0.23141283591588344f;
    const float s3 = -0.076167541742324804f;
    return s1 * in + s2 * in * in + s3 * in * in * in;
}
#elif defined(CORR_POL_0)
static float delta(const float in) {
    static const float kC = 0.05798480f;
    return kC;
}
#endif

static float fast_pow2(const float in) {
    const float yf = in - (int32_t)(in) + 1;
    u_float_int_t value;
    value.i32 = (int32_t)((1 << 23) * (in - delta(yf) + 127.0f));
#if 0
    printf("in % 2.04f | frac % 2.04f |  val %#010x  % 2.04f\r\n", in, yf, value.i32,
           value.f32);
#endif
    return value.f32;
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
