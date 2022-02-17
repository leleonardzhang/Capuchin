/*
 * matrix_ops.c
 * Include matrix operations and especially LEA calls relating to LEA RAM
 */

#include "matrix_ops.h"

static uint16_t SPARSEMAX_BUFFER[NUM_OUTPUTS] = {0};

#ifdef IS_MSP
#include "DSPLib.h"

/*
 *  Allocate memory in the LEA RAM.
 *  This part of memory is used when executing LEA matrix multiplication and element-wise multiplication.
 */
DSPLIB_DATA(MULTIPLY_BUFFER, 4);
static dtype MULTIPLY_BUFFER[LEA_RAM_LENGTH];

matrix *filter_LEA(matrix* result, matrix *input, matrix *filter, uint16_t precision, uint16_t stride_numRows, uint16_t stride_numCols){
    /*
     * LEA Implementation of one filter of a conv2d layer
     * Use the element-wise multiplication && sum function, msp_mpy_q15, provided by DSPLib
     */

    uint16_t input_numRows = input->numRows;
    uint16_t input_numCols = input->numCols;
    uint16_t filter_numRows = filter->numRows;
    uint16_t filter_numCols = filter->numCols;
    uint16_t i, j, m, n, input_temp;

    uint16_t filter_offset;
    uint16_t filter_length = filter_numRows * filter_numCols;
    dtype *resultData = result->data;
    result->numRows = (input_numRows - filter_numRows) / stride_numRows + 1;
    result->numCols = (input_numCols - filter_numCols) / stride_numCols + 1;

    if (filter_length & 1) filter_offset = filter_length + 1; // macParams.length must be a multiple of 2

    /*
     * msp_mac_q15() does element-wise multiplication and then calculate the sum, e.g [1,2,3,4] * [2,2,2,2]  = [2,4,6,8] --> sum = 20
     * MULTIPLY_BUFFER resides in the 4KB LEA RAM
     * srcA is the part of LEA RAM that saves the part of the input where filter will be applied
     * srcB is another part of LEA RAM that saves the kernel of the filter itself
     */
    dtype *srcA = MULTIPLY_BUFFER;
    dtype *srcB = MULTIPLY_BUFFER + filter_offset;
    memset(srcB, 0, sizeof(dtype) * filter_offset);             // set memory to zero to prevent the error of the last element added in order to meet the requirement of multiple of 2
    memcpy(srcB, filter->data, sizeof(dtype) * filter_offset);  // only copy kernel one time for a single channel
    _iq31 *result_iq31 = srcB + filter_offset;                  // msp_mac_q15() returns 32-bit result
    _q15 *result_q15 = result_iq31 + sizeof(result_iq31) * 2;   // msp_iq31_to_q15() returns 16-bit result

    /* for msp_mac_q15() params */
    msp_status status;
    msp_mac_q15_params macParams;
    macParams.length = filter_offset;               // must be a multiple of 2

    /* for msp_iq31_to_q15() params */
    msp_iq31_to_q15_params iq31_to_q15_Params;
    iq31_to_q15_Params.length = 2;                  // must be a multiple of 2

    /* for msp_shift_iq31_params() params */
    msp_shift_iq31_params shiftParams;
    shiftParams.length = 2;                         // must be a multiple of 2
    shiftParams.shift = 15 - precision;

    for (i = 0; i <= input_numRows - filter_numRows; i += stride_numRows){
        for (j = 0; j <= input_numCols - filter_numCols; j += stride_numCols){
            /* (i,j) is the coordinate of the top-left element of the area in the input that will be applied with filter */
            dtype *srcA_temp = srcA;
            for (m = i; m < i + filter_numRows; m ++ ) {
                for (n = j; n < j + filter_numCols; n ++) {
                    /* copy the entire area into srcA inside of LEA RAM */
                    input_temp = m * input_numCols + n;
                    *(srcA_temp++) = input->data[input_temp];
                }
            }

            /* apply element-wise multiplication and sum LEA API */
            status = msp_mac_q15(&macParams, srcA, srcB, result_iq31);
            msp_checkStatus(status);

            /* apply shifting LEA API */
            status = msp_shift_iq31(&shiftParams, result_iq31, result_iq31);
            msp_checkStatus(status);

            /* apply 32-to-16-bit conversion LEA API */
            status = msp_iq31_to_q15(&iq31_to_q15_Params, result_iq31, result_q15);
            msp_checkStatus(status);

            /* save result back to FRAM */
            *resultData = *result_q15;
            resultData ++;
        }
    }
    return result;
}

dtype *dma_load(dtype *result, dtype *data, uint16_t n) {
    /*
     * Need change if read/write 20-bits addresses (FRAM2)
     * Loads the first n elements of the data array into the result array using
     * DMA.
     */
    // Configure DMA channel 0
    __data20_write_long((uintptr_t) &DMA0SA, (uintptr_t) data);   // Source block address
    __data20_write_long((uintptr_t) &DMA0DA, (uintptr_t) result); // Destination single address
    DMA0SZ = n;                                      // Block size
    DMA0CTL = DMADT_5 | DMASRCINCR_3 | DMADSTINCR_3; // Rpt, inc
    DMA0CTL |= DMAEN;                                // Enable DMA0
    DMA0CTL |= DMAREQ;

    return result;
}
#endif

matrix *matrix_add(matrix *result, matrix *mat1, matrix *mat2) {
    /**
     * Adds two matrices together element-wise
     * Result stored stored in the result argument (and returned for convenience).
     */
    // Validate dimensions
    if ((mat1->numRows != mat2->numRows) || (result->numRows != mat1->numRows)) {
        return NULL_PTR;
    }
    
    uint16_t rows = mat1->numRows;
    uint16_t cols = mat1->numCols > mat2->numCols ? mat2->numCols : mat1->numCols;

    uint16_t mat1Offset, mat2Offset, resultOffset;
    uint16_t rowOffset, colOffset;

    // Compute the element-wise sum. We generally add small vectors together, and the LEA
    // provides little (or negative) benefit due to added overhead.
    uint16_t i, j;
    for (i = rows; i > 0; i--) {

        rowOffset = i - 1;

        mat1Offset = rowOffset * mat1->numCols;
        mat2Offset = rowOffset * mat2->numCols;
        resultOffset = rowOffset * result->numCols;

        for (j = cols; j > 0; j--) {
            colOffset = j - 1;
            result->data[resultOffset + colOffset] = fp_add(mat1->data[mat1Offset + colOffset], mat2->data[mat2Offset + colOffset]);
        }
    }

    return result;
}


matrix *matrix_neg(matrix *result, matrix *mat, uint16_t precision) {
    /*
     * Multiplies all elements by -1.
     */
    return scalar_product(result, mat, int_to_fp(-1, precision), precision);
}


matrix *matrix_multiply(matrix *result, matrix *mat1, matrix *mat2, uint16_t precision) {
    /**
     * Performs matrix multiplication and stores value in given result array. The implementation
     * depends on whether or not we are compiling for the MSP430 device.
     */

    // Validate dimensions
    if ((mat1->numCols != mat2->numRows) || (mat1->numRows != result->numRows) || (mat2->numCols != result->numCols)) {
        return NULL_PTR;
    }

    // The result will be a [n, p] matrix
    uint16_t n = mat1->numRows;
    uint16_t m = mat1->numCols;
    uint16_t p = mat2->numCols;

    #ifdef IS_MSP

    if (n * m + m * p + n * p > 1800){
        /*
         * LEA RAM has the limit of 1800 for 16-bit data
         */
        uint16_t i, j, k;
        uint16_t outerRow, innerRow, resultRow;
        int16_t sum, prod;

        for (i = n; i > 0; i--) {
            outerRow = (i - 1) * m;  // Offset for the i^th row

            for (j = p; j > 0; j--) {
                sum = 0;

                for (k = m; k > 0; k--) {
                    innerRow = (k - 1) * p;  // Offset for the k^th row
                    prod = fp_mul(mat1->data[outerRow + (k - 1)], mat2->data[innerRow + (j - 1)], precision);
                    sum = fp_add(sum, prod);
                }

                resultRow = (i - 1) * p;
                result->data[resultRow + (j - 1)] = sum;
            }
        }
    }
    else {
        uint16_t offset = 0;
        dtype *mat1Data = memcpy(MULTIPLY_BUFFER, mat1->data, n * m * sizeof(dtype));
        offset += n * m;

        dtype *mat2Data = memcpy(MULTIPLY_BUFFER + offset, mat2->data, m * p * sizeof(dtype));
        offset += m * p;

        dtype *resultData = MULTIPLY_BUFFER + offset;  // Temporary buffer (in LEA RAM) for the result

        // When using the MSP430, we use the LEA for matrix multiplications. Based on profiling,
        // the LEA can take up to 5x fewer compute cycles than a standard implementation.
        msp_status status;
        msp_matrix_mpy_q15_params mulParams;

        // Initialze LEA metadata
        mulParams.srcARows = n;
        mulParams.srcACols = m;
        mulParams.srcBRows = m;
        mulParams.srcBCols = p;

        // Perform matrix multiplication using the LEA
        status = msp_matrix_mpy_q15(&mulParams, mat1Data, mat2Data, resultData);
        msp_checkStatus(status);

        // Convert back to the original fixed-point precision. The LEA assumes 15 fractional bits.
        msp_matrix_shift_q15_params shiftParams;
        shiftParams.rows = n;
        shiftParams.cols = p;
        shiftParams.shift = 15 - precision;

        // Perform element-wise shift using the LEA
        if (shiftParams.shift > 0) {
            status = msp_matrix_shift_q15(&shiftParams, resultData, resultData);
            msp_checkStatus(status);
        }

        // Load result back into the given result matrix
        memcpy(result->data, resultData, n * p * sizeof(dtype));
    }

    #else

    uint16_t i, j, k;
    uint16_t outerRow, innerRow, resultRow;
    int16_t sum, prod;

    for (i = n; i > 0; i--) {
        outerRow = (i - 1) * m;  // Offset for the i^th row

        for (j = p; j > 0; j--) {
            sum = 0;

            for (k = m; k > 0; k--) {
                innerRow = (k - 1) * p;  // Offset for the k^th row
                prod = fp_mul(mat1->data[outerRow + (k - 1)], mat2->data[innerRow + (j - 1)], precision);
                sum = fp_add(sum, prod);
            }
 
            resultRow = (i - 1) * p;
            result->data[resultRow + (j - 1)] = sum;
        }
    }
    #endif

    return result;
}


int16_t dot_product(matrix *vec1, matrix *vec2, uint16_t precision) {
    /**
     * Computes the dot product for the two vectors. If the inputs are not
     * proper vectors, then we use the first row of vec1 and first column of vec2.
     */
    uint16_t i;
    uint16_t vec1Idx, vec2Idx;
    int16_t result = 0;

    for (i = vec1->numCols; i > 0; i--) {
        vec1Idx = i - 1;
        vec2Idx = vec2->numCols * (i - 1);

        result = fp_add(result, fp_mul(vec1->data[vec1Idx], vec2->data[vec2Idx], precision));
    }

    return result;
}



matrix *matrix_hadamard(matrix* result, matrix *mat1, matrix *mat2, uint16_t precision) {
    /**
     * Elementwise matrix product. Result stored in matrix 1. We never use the LEA to avoid
     * added overhead (we only multiply small vectors).
     */
    // Validate dimensions
    if ((mat1->numRows != mat2->numRows) || (result->numRows != mat1->numRows)) {
        return NULL_PTR;
    }
    
    uint16_t rows = mat1->numRows;
    uint16_t cols = mat1->numCols > mat2->numCols ? mat2->numCols : mat1->numCols;

    uint16_t mat1Offset, mat2Offset, resultOffset;
    uint16_t rowOffset, colOffset;

    // Compute the element-wise sum. We generally add small vectors together, and the LEA
    // provides little (or negative) benefit due to added overhead.
    uint16_t i, j;
    for (i = rows; i > 0; i--) {

        rowOffset = i - 1;

        mat1Offset = rowOffset * mat1->numCols;
        mat2Offset = rowOffset * mat2->numCols;
        resultOffset = rowOffset * result->numCols;

        for (j = cols; j > 0; j--) {
            colOffset = j - 1;
            result->data[resultOffset + colOffset] = fp_mul(mat1->data[mat1Offset + colOffset], mat2->data[mat2Offset + colOffset], precision);
        }
    }

    return result;
}


matrix *scalar_product(matrix *result, matrix *mat, int16_t scalar, uint16_t precision) {
    /**
     * Multiplies every element in the matrix by the given scalar.
     * Result stored directly into the given array.
     */
    // Validate dimensions
    if ((result->numRows != mat->numRows) || (result->numCols != mat->numCols)) {
        return NULL_PTR;
    }

    uint16_t i;
    for (i = mat->numRows * mat->numCols; i > 0; i--) {
        result->data[i - 1] = fp_mul(mat->data[i - 1], scalar, precision);
    }

    return result;
}


matrix *scalar_add(matrix *result, matrix *mat, int16_t scalar) {
    /**
     * Adds the given scalar to every element of the matrix.
     * Stores result directly in the matrix.
     */
    // Validate dimensions
    if ((result->numRows != mat->numRows) || (result->numCols != mat->numCols)) {
        return NULL_PTR;
    }

    uint16_t i;
    for (i = mat->numRows * mat->numCols; i > 0; i--) {
        result->data[i - 1] = fp_add(mat->data[i - 1], scalar);
    }
    
    return result;
}


matrix *apply_elementwise(matrix *result, matrix *mat, int16_t (*fn)(int16_t, uint16_t), uint16_t precision) {
    /**
     * Applies the given function to every element of the
     * input matrix. Result stored directly in the matrix.
     */
    // Validate dimensions
    if ((result->numRows != mat->numRows) || (result->numCols != mat->numCols)) {
        return NULL_PTR;
    }

    uint16_t i;
    for (i = mat->numRows * mat->numCols; i > 0; i--) {
        result->data[i - 1] = (*fn)(mat->data[i - 1], precision);
    }

    return result;
}


matrix *matrix_replace(matrix *dst, matrix *src) {
    /**
     * Replaces the contents of the destination matrix with those from the src.
     */
    if ((dst->numRows != src->numRows) || (dst->numCols != src->numCols)) {
        return NULL_PTR;
    }

    uint16_t i;
    for (i = dst->numRows * dst->numCols; i > 0; i--) {
        dst->data[i - 1] = src->data[i - 1];
    }

    return dst;
}


matrix *matrix_set(matrix *mat, int16_t value) {
    /**
     * Sets all values in the matrix to the given value (already in fixed point form).
     */

    uint16_t i;
    for (i = mat->numRows * mat->numCols; i > 0; i--) {
        mat->data[i - 1] = value;
    }

    return mat;
}


int16_t matrix_sum(matrix *mat) {
    /**
     * Computes the sum of all elements in the matrix
     */
    int16_t sum = 0;
    uint16_t i;
    for (i = mat->numRows * mat->numCols; i > 0; i--) {
        sum = fp_add(mat->data[i - 1], sum);
    }

    return sum;
}


int16_t matrix_min(matrix *mat) {
    /**
     * Computes the minimum value over all elements in the matrix
     */
    int16_t min_value = 32767;  // 2^15 - 1
    int16_t val;
    uint16_t i;
    for (i = mat->numRows * mat->numCols; i > 0; i--) {
        val = mat->data[i - 1];
        if (val < min_value) {
            min_value = val;
        }
    }

    return min_value;
}



matrix *vstack(matrix *result, matrix *mat1, matrix *mat2) {
    /**
     * Stacks the two matrices into a larger matrix. For example, if mat1 is [n, m] and
     * mat2 is [p, m], the result must be [n + p, m] where mat is placed about mat2.
     */
    // Validate the input shapes.
    if ((mat1->numRows + mat2->numRows != result->numRows) || (mat1->numCols != mat2->numCols) ||
        (mat1->numCols != result->numCols)) {
        return NULL_PTR;
    }

    uint16_t cols = mat1->numCols;

    // Copy in the first matrix
    uint16_t i;
    for (i = mat1->numRows * cols; i > 0; i--) {
        result->data[i-1] = mat1->data[i-1];
    }

    // Copy in the second matrix
    uint16_t offset = mat1->numRows * cols;
    for (i = mat2->numRows * cols; i > 0; i--) {
        result->data[offset + i - 1] = mat2->data[i-1];
    }

    return result;
}


int16_t argmax(matrix *vec) {
    /**
     * Computes the argmax of the 1d vector. If the input has multiple columns, then
     * this is the argmax over the 1st column.
     */
    if (vec->numRows <= 0) {
        return -1;
    }

    uint16_t numCols = vec->numCols;

    int16_t max = vec->data[0];
    int16_t max_index = 0;

    uint16_t i;
    int16_t val;
    for (i = vec->numRows - 1; i > 0; i--) {
        val = vec->data[i * numCols];
        if (val > max) {
            max_index = i;
            max = val;
        }
    }

    return max_index;
}


uint16_t *argsort(matrix *vec, uint16_t *result) {
    // Sorts indices in descending order
    // according to the first column of the given matrix. Sorting
    // performed by insertion sort, as the vectors are generally small.

    // Initialize the result indices.
    uint16_t i;
    for (i = 0; i < vec->numRows; i++) {
        result[i] = i * vec->numCols;
    }

    uint16_t j, k;
    uint16_t idx1, idx2;
    int16_t t;

    for (k = vec->numRows; k > 0; k--) {
        i = vec->numRows - k;

        for (j = i; j > 0; j--) {
            idx1 = result[j-1];
            idx2 = result[j];

            // Swap if result[j] corresponds to a larger
            // value than result[j-1]
            if (vec->data[idx2] > vec->data[idx1]) {
                t = result[j-1];
                result[j-1] = result[j];
                result[j] = t;
            }
        }
    }

    return result;
}


matrix *sparsemax(matrix *result, matrix *vec, uint16_t precision) {
    // Sort indices of the vector in descending order
    uint16_t *sortedIndices = SPARSEMAX_BUFFER;
    argsort(vec, sortedIndices);

    // Compute the k(z) function
    int16_t partialSum = 0;
    int16_t one = 1 << precision;
    int16_t zk = 0;
    int16_t coordinate = 0;
    uint16_t k = 0;

    uint16_t i;
    for (i = vec->numRows; i > 0; i--) {
        k = vec->numRows - i + 1;
        zk = vec->data[sortedIndices[k - 1]];
        partialSum = fp_add(partialSum, zk);

        coordinate = fp_add(one, fp_mul(int_to_fp(k, precision), zk, precision));

        if (coordinate <= partialSum) {
            k = k - 1;
            partialSum = fp_add(partialSum, fp_neg(zk));
            break;
        }
    }

    // Compute the threshold, t(z)
    int16_t kz = int_to_fp(k, precision);
    int16_t sumMinusOne = fp_add(partialSum, fp_neg(one));
    int16_t threshold = fp_div(sumMinusOne, kz, precision);

    // Use the threshold to apply the sparse normalization function
    uint16_t j, idx;
    int16_t diff;
    for (j = vec->numRows; j > 0; j--) {
        idx = (j - 1) * vec->numCols;
        diff = fp_sub(vec->data[idx], threshold);
        result->data[idx] = fp_relu(diff, precision);
    }

    return result;
}
