#include "fixed_point_ops.h"


int16_t fp_add(int16_t x, int16_t y) {
    return x + y;
}


int16_t fp_sub(int16_t x, int16_t y) {
    return x - y;
}


int16_t fp_mul(int16_t x, int16_t y, uint16_t precision) {
    int32_t mul = ((int32_t) x) * ((int32_t) y);
    return (int16_t) (mul >> precision);
}


int16_t fp_div(int16_t x, int16_t y, uint16_t precision) {
    int32_t xLarge = ((int32_t) x) << precision;
    return (int16_t) (xLarge / y);
}


int16_t fp_neg(int16_t x) {
    return -1 * x;
}


int16_t fp_mod(int16_t x, int16_t m, uint16_t precision) {
    int16_t div = fp_div(x, m, precision);
    int16_t floorDiv = div & ~((1 << precision) - 1);
    return fp_add(x, fp_neg(fp_mul(floorDiv, m, precision)));
}


int16_t convert_fp(int16_t x, uint16_t old_precision, uint16_t new_precision) {
    return (x * (1 << new_precision)) / (1 << old_precision);
}


int16_t float_to_fp(float x, uint16_t precision) {
    return (int16_t) (x * (1 << precision));
}


int16_t int_to_fp(int16_t x, uint16_t precision) {
    return x * (1 << precision);
}


int16_t fp_round_to_int(int16_t x, uint16_t precision) {
    int8_t should_invert_sign = 0;
    if (x < 0) {
        should_invert_sign = 1;
        x = fp_neg(x);
    }

    int16_t fractionMask = (1 << precision) - 1;
    int16_t fractionalPart = x & fractionMask;
    int16_t integerPart = x & ~(fractionMask);

    int16_t roundedVal;
    int16_t one_half = 1 << (precision - 1);
    if (fractionalPart >= one_half) {
        roundedVal = fp_add(integerPart, int_to_fp(1, precision));
    } else {
        roundedVal = integerPart;
    }

    if (should_invert_sign) {
        return fp_neg(roundedVal);
    }
    return roundedVal;
}


int16_t fp_relu(int16_t x, uint16_t precision) {
    UNUSED(precision);
    if (x >= 0) {
        return x;
    }
    return 0;
}


int16_t fp_leaky_relu(int16_t x, uint16_t precision) {
    UNUSED(precision);
    int16_t isPositive = (int16_t) (x > 0);

    // We perform division by 4 like this because bit shifting
    // is more efficient than division on the MSP430
    int16_t leakyX = (x >> 2);
    return isPositive * x + (1 - isPositive) * leakyX;
}


int16_t fp_linear(int16_t x, uint16_t precision) {
    UNUSED(precision);
    return x;
}


int16_t fp_tanh(int16_t x, uint16_t precision) {
    /**
     * Approximates tanh using a polynomial.
     */
    int16_t should_invert_sign = 0;
    if (x < 0) {
        x = fp_neg(x);
        should_invert_sign = 1;
    }

    // Approximate tanh(x) as x * (0.5 + (x/4)^2)/(0.5 + (x/2)^2)
    // Dividing first helps mitigate overflow.
    int16_t one_fourth = 1 << (precision - 2);
    int16_t one_half = 1 << (precision - 1);
    int16_t one = int_to_fp(1, precision);

    int16_t half_x = fp_mul(x, one_half, precision);
    int16_t quarter_x = fp_mul(x, one_fourth, precision);

    int16_t numerator = fp_add(one_half, fp_mul(quarter_x, quarter_x, precision));
    int16_t denominator = fp_add(one_half, fp_mul(half_x, half_x, precision));

    int16_t rational_factor = fp_div(numerator, denominator, precision);
    int16_t result = fp_mul(x, rational_factor, precision);

    if (should_invert_sign) {
        result = fp_neg(result);
    }

    // Clip the output
    int16_t neg_one = fp_neg(one);
    if (result > one) {
        return one;    
    } else if (result < neg_one) {
        return neg_one;
    }

    return result;
}


int16_t fp_sigmoid(int16_t x, uint16_t precision) {
    /**
     * Approximates the sigmoid function using tanh.
     */
    uint8_t should_invert_sign = 0;
    if (x < 0) {
        x = fp_neg(x);
        should_invert_sign = 1;
    }

    int16_t one = 1 << precision;
    int16_t one_half = 1 << (precision - 1);

    int16_t half_x = fp_mul(x, one_half, precision);
    int16_t tanh = fp_tanh(half_x, precision);
    int16_t result = fp_mul(fp_add(tanh, one), one_half, precision);

    if (should_invert_sign) {
        result = one - result;
    }

    return result;
}


// 32 bit fixed-point operations
int32_t fp32_add(int32_t x, int32_t y) {
    return x + y;
}


int32_t fp32_neg(int32_t x) {
    return -1 * x;
}


int32_t fp32_sub(int32_t x, int32_t y) {
    return fp32_add(x, fp32_neg(y));
}


int32_t fp32_mul(int32_t x, int32_t y, uint16_t precision) {
    int64_t xLarge = (int64_t) x;
    int64_t yLarge = (int64_t) y;

    return (int32_t) ((xLarge * yLarge) >> precision);
}


int32_t fp32_div(int32_t x, int32_t y, uint16_t precision) {
    int64_t xLarge = ((int64_t) x) << precision;
    return (int32_t) (xLarge / y);
}


int32_t fp32_sqrt(int32_t x, uint16_t precision) {
    if (x < 0) {
        return 0;
    }

    int32_t hundred = int_to_fp32(100, precision);
//    int32_t ten = int_to_fp32(10, precision);
    volatile int32_t a = x;
    volatile int32_t n = 1;

    while (a > hundred) {
//        a = fp32_div(a, hundred, precision);
//        n = fp32_mul(n, ten, precision);
        a /= 100;
        n *= 10;
    }

    const int32_t one256 = 1 << (precision - 8);
    const int32_t one32 = 1 << (precision - 5);
    const int32_t oneEight = 1 << (precision - 3);
    const int32_t oneHalf = 1 << (precision - 1);
    const int32_t two = 1 << (precision + 1);
    const int32_t eight = 1 << (precision + 3);
    const int32_t thirtyTwo = ((int32_t) 1) << (precision + 5);

    volatile int32_t sqrtPart = 0;

    if (a <= one256) {
        sqrtPart = fp32_add(a << 4, 1 << (precision - 6));  // 16x + 1/64
    } else if (a <= one32) {
        sqrtPart = fp32_add(a << 2, 1 << (precision - 5));  // 4x + 1/16
    } else if (a <= oneEight) {
        sqrtPart = fp32_add(a << 1, 1 << (precision - 3));  // 2x + 1/8
    } else if (a <= oneHalf) {
        sqrtPart = fp32_add(a, 1 << (precision - 2));  // x + 1/4
    } else if (a <= two) {
        sqrtPart = fp32_add(a >> 1, (int32_t) oneHalf);  // x/2 + 1/2
    } else if (a <= eight) {
        sqrtPart = fp32_add(a >> 2, 1 << precision);  // x/4 + 1
    } else if (a <= thirtyTwo) {
        sqrtPart = fp32_add(a >> 3, 1 << (precision + 1));  // x/8  + 2
    } else {
        sqrtPart = fp32_add(a >> 4, 1 << (precision + 2));  // x/16 + 4
    }

    return sqrtPart * n;
    // return fp32_mul(sqrtPart, n, precision);
}


int32_t int_to_fp32(int32_t x, uint16_t precision) {
    return ((int32_t) x) << precision;
}
