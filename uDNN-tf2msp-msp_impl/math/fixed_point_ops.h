#include <stdint.h>
#include "../utils/utils.h"

#ifndef FIXED_POINT_OPS_GUARD
#define FIXED_POINT_OPS_GUARD

    int16_t fp_add(int16_t x, int16_t y);
    int16_t fp_mul(int16_t x, int16_t y, uint16_t precision);
    int16_t fp_sub(int16_t x, int16_t y);
    int16_t fp_div(int16_t x, int16_t y, uint16_t precision);
    int16_t fp_neg(int16_t x);
    int16_t fp_mod(int16_t x, int16_t m, uint16_t precision);
    int16_t fp_tanh(int16_t x, uint16_t precision);
    int16_t fp_sigmoid(int16_t x, uint16_t precision);
    int16_t fp_relu(int16_t x, uint16_t precision);
    int16_t fp_leaky_relu(int16_t x, uint16_t precision);
    int16_t fp_linear(int16_t x, uint16_t precision);
    int16_t fp_round_to_int(int16_t x, uint16_t precision);
    int16_t convert_fp(int16_t x, uint16_t old_precision, uint16_t new_precision);
    int16_t float_to_fp(float x, uint16_t precision);
    int16_t int_to_fp(int16_t x, uint16_t precision);

    // 32 bit fixed point operations for improved precision. These are slightly more expensive
    // on 16 bit MCUs
    int32_t fp32_add(int32_t x, int32_t y);
    int32_t fp32_neg(int32_t x);
    int32_t fp32_sub(int32_t x, int32_t y);
    int32_t fp32_mul(int32_t x, int32_t y, uint16_t precision);
    int32_t fp32_div(int32_t x, int32_t y, uint16_t precision);
    int32_t fp32_sqrt(int32_t x, uint16_t precision);
    int32_t int_to_fp32(int32_t x, uint16_t precision);

#endif
