#include "rvv_softmax_f32.h"

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
        vfloat32m8_t vNext = __riscv_vle32_v_f32m8(cur_pos, vl);
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

static vfloat32m8_t rvv_get_frac(vfloat32m8_t vIn, size_t vl) {
    vint32m8_t vInt;
    vInt = __riscv_vfcvt_rtz_x_tu(vInt, vIn, vl);
    vfloat32m8_t vResult, vIntF;
    vIntF = __riscv_vfcvt_f_x_v_f32m8(vInt, vl);
    vResult = __riscv_vfsub_vv_f32m8(vIn, vIntF, vl);
    return vResult;
}

static vfloat32m8_t rvv_delta(vfloat32m8_t vIn, size_t vl) {
    const float s1 = 0.30758037765820823f;
    const float s2 = -0.23141283591588344f;
    const float s3 = -7.6167541742324804e-2f;
    vfloat32m8_t vResult, vP1, vP2, vP3, vS1, vS2, vS3;
    vS1 = __riscv_vfmv_v_f_f32m8(s1, vl);
    vS2 = __riscv_vfmv_v_f_f32m8(s2, vl);
    vS3 = __riscv_vfmv_v_f_f32m8(s3, vl);

    vP1 = __riscv_vfmul_vv_f32m8_tu(vP1, vIn, vS1, vl);

    vP2 = __riscv_vfmul_vv_f32m8_tu(vP2, vIn, vIn, vl);
    vP2 = __riscv_vfmul_vv_f32m8_tu(vP2, vP2, vS2, vl);

    vP3 = __riscv_vfmul_vv_f32m8_tu(vP3, vIn, vIn, vl);
    vP3 = __riscv_vfmul_vv_f32m8_tu(vP3, vP3, vIn, vl);
    vP3 = __riscv_vfmul_vv_f32m8_tu(vP3, vP3, vS3, vl);

    vResult = __riscv_vfadd_vv_f32m8(vP1, vP2, vl);
    vResult = __riscv_vfadd_vv_f32m8(vResult, vP3, vl);

    return vResult;
}


static vfloat32m8_t rvv_fast_pow2(vfloat32m8_t vIn, size_t vl) {
    vfloat32m8_t vFrac = rvv_get_frac(vIn, vl);
    vfloat32m8_t vDelta = rvv_delta(vFrac, vl);

    vfloat32m8_t vTmp;
    vTmp = __riscv_vfsub_vv_f32m8(vIn, vDelta, vl);
    vTmp = __riscv_vfadd_vf_f32m8(vTmp, vTmp, 127.0f, vl);
    return vTmp;
}

// Softmax Functions
int32_t rvv_softmax_f32(const float *in_vec, uint32_t size, float *out_vec) {
    float max = rvv_find_max(in_vec, size);

    // calculate the exp and sum
    // vfloat32m8_t vSum = __riscv_vfmv_v_f_f32m8(0.0f, vl);

    // in_vec2 = (float *)in_vec;
    // out_vec2 = out_vec;
    // size2 = size;

    // while (size2 > 0) {
    //     vl = __riscv_vsetvl_e32m8(size2);
    //     vfloat32m8_t vData = __riscv_vle32_v_f32m8(in_vec2, vl);

    //     in_vec2 += vl;
    //     vData = __riscv_vfsub_vf_f32m8(vData, max, vl);
    //     size2 -= vl;

    //     //--- exp_f32 begin ---
    //     vData = ace_exp_f32m8(vData, vl);
    //     //--- exp_f32 end ---

    //     vSum = __riscv_vfadd_vv_f32m8(vSum, vData, vl);  // accumulate
    //     sum_f32
    //     __riscv_vse32_v_f32m8(out_vec2, vData, vl);

    //     out_vec2 += vl;
    // }

    // vl = __riscv_vsetvl_e32m1(1);
    // vfloat32m1_t vResult = __riscv_vfmv_v_f_f32m1(0.0f, vl);
    // vl = __riscv_vsetvl_e32m8(size);
    // vResult = __riscv_vfredosum_vs_f32m8_f32m1(vSum, vResult, vl);
    // sum = __riscv_vfmv_f_s_f32m1_f32(vResult);

    // // divide the output by sum
    // size2 = size;

    // while (size2 > 0) {
    //     vl = __riscv_vsetvl_e32m8(size2);
    //     vfloat32m8_t vData = __riscv_vle32_v_f32m8(out_vec, vl);
    //     vData = __riscv_vfdiv_vf_f32m8(vData, sum, vl);
    //     __riscv_vse32_v_f32m8(out_vec, vData, vl);
    //     out_vec += vl;
    //     size2 -= vl;
    // }

    return 0;
}