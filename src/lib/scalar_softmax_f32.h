#pragma once

#include <stdint.h>

int32_t scalar_softmax_f32(const float* in_vec, uint32_t size, float* out_vec);
