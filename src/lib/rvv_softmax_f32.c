

#include <riscv_vector.h>

static float rvv_find_max(const float *in_vec, uint32_t size) {
    float *cur_pos = (float *)in_vec;
    size_t rem_size = size;
    size_t vl = __riscv_vsetvl_e32m8(rem_size);
    vfloat32m8_t vInput = __riscv_vle32_v_f32m8(cur_pos, vl);

    cur_pos += vl;
    rem_size -= vl;
    while (rem_size > 0) {
        vl = __riscv_vsetvl_e32m8(rem_size);
        const vfloat32m8_t vNext = __riscv_vle32_v_f32m8(cur_pos, vl);
        cur_pos += vl;
        rem_size -= vl;
        vInput = __riscv_vfmax_vv_f32m8(vInput, vNext, vl);
    }

    vl = __riscv_vsetvl_e32m1(size);
    vfloat32m1_t vMax = __riscv_vfmv_v_f_f32m1(0.0f, vl);

    vl = __riscv_vsetvl_e32m8(size);
    vMax = __riscv_vfredmax_vs_f32m8_f32m1(vInput, vMax, vl);
    float max = __riscv_vfmv_f_s_f32m1_f32(vMax);
    return max;
}

#define USE_ACE
// Softmax Functions
#ifdef USE_ACE
#include "ace_user.h"
//vfloat32m8_t ace_exp_f32m8(vfloat32m8_t x, size_t vl);
int32_t rvv_ace_softmax_f32(const float *in_vec, uint32_t size,
                            float *out_vec) {
    float max = rvv_find_max(in_vec, size);

    float *cur_pos = (float *)in_vec;
    float *cur_out_pos = (float *)out_vec;
    size_t rem_size = size;

    size_t vl = __riscv_vsetvl_e32m8(rem_size);

    vfloat32m8_t vSum = __riscv_vfmv_v_f_f32m8(0.0f, vl);

    while (rem_size > 0) {
        vl = __riscv_vsetvl_e32m8(rem_size);
        vfloat32m8_t vData = __riscv_vle32_v_f32m8(cur_pos, vl);

        cur_pos += vl;
        rem_size -= vl;

        // Substract max
        vData = __riscv_vfsub_vf_f32m8(vData, max, vl);
        //exp_f32
        vData = ace_exp_f32m8(vData, vl);

        vSum = __riscv_vfadd_vv_f32m8(vSum, vData, vl);  // accumulate
        // save value to memory
        __riscv_vse32_v_f32m8(cur_out_pos, vData, vl);

        cur_out_pos += vl;
    }

    vl = __riscv_vsetvl_e32m1(1);
    vfloat32m1_t vResult = __riscv_vfmv_v_f_f32m1(0.0f, vl);
    vl = __riscv_vsetvl_e32m8(size);
    vResult = __riscv_vfredosum_vs_f32m8_f32m1(vSum, vResult, vl);
    float sum = __riscv_vfmv_f_s_f32m1_f32(vResult);

    // divide the output by sum
    rem_size = size;

    while (rem_size > 0) {
        vl = __riscv_vsetvl_e32m8(rem_size);
        vfloat32m8_t vData = __riscv_vle32_v_f32m8(out_vec, vl);
        vData = __riscv_vfdiv_vf_f32m8(vData, sum, vl);
        __riscv_vse32_v_f32m8(out_vec, vData, vl);
        out_vec += vl;
        rem_size -= vl;
    }

    return 0;
}
#else
// #define SIZE 3
// #include <stdio.h>
static const float k1OverLn2 = 1.44269504089f;


static vfloat32m8_t rvv_get_frac(const vfloat32m8_t vIn, size_t vl) {
    vint32m8_t vInt = __riscv_vmv_v_x_i32m8(0, vl);
    vInt = __riscv_vfcvt_rtz_x_tu(vInt, vIn, vl);
    const vfloat32m8_t vIntF = __riscv_vfcvt_f_x_v_f32m8(vInt, vl);
    vfloat32m8_t vResult = __riscv_vfsub_vv_f32m8(vIn, vIntF, vl);
    vResult = __riscv_vfadd_vf_f32m8(vResult, 1.0f, vl);
    return vResult;
}

static vfloat32m8_t rvv_delta(const vfloat32m8_t vIn, size_t vl) {
    const float s0 = 0.0f;
    const float s1 = 3.06852819440055e-1f;
    const float s2 = -2.40226506959101e-1f;
    const float s3 = -5.57129652016652e-2f;
    const float s4 = -9.01146535969578e-3f;
    const float s5 = -1.90188191959304e-3f;
    vfloat32m8_t vResult = __riscv_vfmv_v_f_f32m8(s5, vl);
    const vfloat32m8_t vS0 = __riscv_vfmv_v_f_f32m8(s0, vl);
    const vfloat32m8_t vS1 = __riscv_vfmv_v_f_f32m8(s1, vl);
    const vfloat32m8_t vS2 = __riscv_vfmv_v_f_f32m8(s2, vl);
    const vfloat32m8_t vS3 = __riscv_vfmv_v_f_f32m8(s3, vl);
    const vfloat32m8_t vS4 = __riscv_vfmv_v_f_f32m8(s4, vl);

    // x = s5
    vResult = __riscv_vfmadd(vResult, vIn, vS4, vl);
    // s5*x + s4
    vResult = __riscv_vfmadd(vResult, vIn, vS3, vl);
    // (s5*x + s4) * x + s3 = s5 x^2 + s4 x + s3
    vResult = __riscv_vfmadd(vResult, vIn, vS2, vl);
    // (s5 x^2 + s4 x + s3) * x + s2 = s5 x^3 + s4 x^2 + s3 x + s2
    vResult = __riscv_vfmadd(vResult, vIn, vS1, vl);
    // (s5 x^3 + s4 x^2 + s3 x + s2) * x + s1 =
    // = s5 x^4 + s4 x^3 + s3 x^2 + s2 x + s1
    vResult = __riscv_vfmadd(vResult, vIn, vS0, vl);
    // (s5 x^4 + s4 x^3 + s3 x^2 + s2 x + s1) * x + s0 =
    // = s5 x^5 + s4 x^4 + s3 x^3+ s2 x^2 + s1 x

    return vResult;
}

static vfloat32m8_t rvv_fast_pow2(const vfloat32m8_t vIn, size_t vl) {
    const vfloat32m8_t vFrac = rvv_get_frac(vIn, vl);
    const vfloat32m8_t vDelta = rvv_delta(vFrac, vl);
    const int32_t pow2_23 = 1 << 23;

    vfloat32m8_t vTmp;
    vTmp = __riscv_vfsub_vv_f32m8(vIn, vDelta, vl);
    vTmp = __riscv_vfadd_vf_f32m8(vTmp, 127.0f, vl);
    vTmp = __riscv_vfmul_vf_f32m8(vTmp, pow2_23, vl);
    vint32m8_t vTmp_i32 = __riscv_vmv_v_x_i32m8(0, vl);
    vTmp_i32 = __riscv_vfcvt_rtz_x_tu(vTmp_i32, vTmp, vl);
    const vfloat32m8_t vResult = __riscv_vreinterpret_v_i32m8_f32m8(vTmp_i32);
    return vResult;
}

int32_t rvv_pure_softmax_f32(const float *in_vec, uint32_t size,
                             float *out_vec) {
    // {
    //     float data[SIZE] = {-0.3501, 0, -0.3015};
    //     float result[SIZE] = {0};
    //     size_t vl = __riscv_vsetvl_e32m8(3);
    //     vfloat32m8_t vData = __riscv_vle32_v_f32m8(data, vl);
    //     vData = rvv_get_frac(vData, vl);
    //     __riscv_vse32_v_f32m8(result, vData, vl);
    //     for (int i = 0; i < SIZE; ++i) {
    //         printf("in % 2.04f | frac % 2.04f\r\n", data[i], result[i]);
    //     }
    // }

    float max = rvv_find_max(in_vec, size);

    float *cur_pos = (float *)in_vec;
    float *cur_out_pos = (float *)out_vec;
    size_t rem_size = size;

    size_t vl = __riscv_vsetvl_e32m8(rem_size);

    vfloat32m8_t vSum = __riscv_vfmv_v_f_f32m8(0.0f, vl);

    while (rem_size > 0) {
        vl = __riscv_vsetvl_e32m8(rem_size);
        vfloat32m8_t vData = __riscv_vle32_v_f32m8(cur_pos, vl);

        cur_pos += vl;
        rem_size -= vl;
        // substract max and scale
        vData = __riscv_vfsub_vf_f32m8(vData, max, vl);
        vData = __riscv_vfmul_vf_f32m8(vData, k1OverLn2, vl);

        //--- pow2_f32 begin ---
        vData = rvv_fast_pow2(vData, vl);
        //--- pow2_f32 end ---

        vSum = __riscv_vfadd_vv_f32m8(vSum, vData, vl);  // accumulate

        // save value to memory
        __riscv_vse32_v_f32m8(cur_out_pos, vData, vl);

        cur_out_pos += vl;
    }

    vl = __riscv_vsetvl_e32m1(1);
    vfloat32m1_t vResult = __riscv_vfmv_v_f_f32m1(0.0f, vl);
    vl = __riscv_vsetvl_e32m8(size);
    vResult = __riscv_vfredosum_vs_f32m8_f32m1(vSum, vResult, vl);
    float sum = __riscv_vfmv_f_s_f32m1_f32(vResult);

    // divide the output by sum
    rem_size = size;

    while (rem_size > 0) {
        vl = __riscv_vsetvl_e32m8(rem_size);
        vfloat32m8_t vData = __riscv_vle32_v_f32m8(out_vec, vl);
        vData = __riscv_vfdiv_vf_f32m8(vData, sum, vl);
        __riscv_vse32_v_f32m8(out_vec, vData, vl);
        out_vec += vl;
        rem_size -= vl;
    }

    return 0;
}
#endif

int32_t rvv_softmax_f32(const float *in_vec, uint32_t size, float *out_vec) {
#ifdef USE_ACE
    return rvv_ace_softmax_f32(in_vec, size, out_vec);
#else
    return rvv_pure_softmax_f32(in_vec, size, out_vec);
#endif
}