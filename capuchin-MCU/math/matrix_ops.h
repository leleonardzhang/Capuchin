#include <stdint.h>
#include "matrix.h"
#include "fixed_point_ops.h"
#include "../utils/utils.h"
#include "../neural_network_parameters.h"

// Imports when compiling for the MSP430 device
#ifdef IS_MSP
#include <msp430.h>
#include "DSPLib.h"
#endif

#ifndef MATRIX_OPS_GUARD
#define MATRIX_OPS_GUARD

#define VECTOR_COLUMN(X)    ((X) * VECTOR_COLS)


//// For MSP implementations, we allocate memory in the LEA RAM.
//// This memory is used when executing matrix multiplications.
//DSPLIB_DATA(MULTIPLY_BUFFER, 4);
//static dtype MULTIPLY_BUFFER[1600];

matrix *filter_LEA(matrix* result, matrix *input, matrix *filter, uint16_t precision, uint16_t stride_numRows, uint16_t stride_numCols);
// Standard matrix operations
matrix *matrix_add(matrix *result, matrix *mat1, matrix *mat2);
matrix *matrix_multiply(matrix *result, matrix *mat1, matrix *mat2, uint16_t precision);
matrix *matrix_hadamard(matrix *result, matrix *mat1, matrix *mat2, uint16_t precision);
matrix *matrix_neg(matrix *result, matrix *mat, uint16_t precision);
matrix *scalar_product(matrix *result, matrix *mat, int16_t scalar, uint16_t precision);
matrix *scalar_add(matrix *result, matrix *mat, int16_t scalar);
matrix *apply_elementwise(matrix *result, matrix *mat, int16_t (*fn)(int16_t, uint16_t), uint16_t precision);
matrix *matrix_set(matrix *mat, int16_t value);
matrix *matrix_replace(matrix *dst, matrix *src);
matrix *vstack(matrix *result, matrix *mat1, matrix *mat2);
int16_t dot_product(matrix *vec1, matrix *vec2, uint16_t precision);
uint16_t *argsort(matrix *vec, uint16_t *result);
matrix *sparsemax(matrix *result, matrix *vec, uint16_t precision);

// Operations useful for various neural network functions
int16_t argmax(matrix *vec);
int16_t matrix_sum(matrix *mat);
int16_t matrix_min(matrix *mat);

#endif
