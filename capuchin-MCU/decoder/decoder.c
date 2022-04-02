/*
 * decoder.c
 * This file decodes the model array encoded by encoder.py and translate the data into the configuration of layers
 */

#include "decoder.h"

/* used to save the results of each layer and prevent data overwrite */
#pragma LOCATION(MODEL_ARRAY_OUTPUT, 0x10000)
#pragma PERSISTENT(MODEL_ARRAY_OUTPUT)
static int16_t MODEL_ARRAY_OUTPUT[MODEL_ARRAY_OUTPUT_LENGTH] = {0};
#pragma PERSISTENT(MODEL_ARRAY_TEMP)
static int16_t MODEL_ARRAY_TEMP[MODEL_ARRAY_TEMP_LENGTH] = {0};

matrix *apply_model(matrix *output, matrix *input){

    uint16_t i = 0;
    int16_t *array = MODEL_ARRAY;
    int16_t *bias_array;

    uint16_t array_length = MODEL_ARRAY_LENGTH;

    uint16_t layer_class, activation, numChannels, filter_numRows, filter_numCols, stride_numRows, stride_numCols, filters_length, padding;
    uint16_t numFilters;
    output->data = MODEL_ARRAY_OUTPUT;

    i = 0;
    // Sequential model
    if (array[i] == 0){  // 1st element of the array tells the model type
        i ++;
        while (i != array_length){
            // next element of the array tells the layer class

            /* layer class 0 - DENSE */
            if (array[i] == DENSE_LAYER){
                numFilters = 1;

                // extract and prepare layer parameters
                layer_class = array[i];
                activation = array[i+1];
                uint16_t kernel_numRows = array[i+2];
                uint16_t kernel_numCols = array[i+3];
                uint16_t bias_numRows = array[i+4];
                uint16_t bias_numCols = array[i+5];
                i += 6;
                uint16_t kernel_length = kernel_numRows * kernel_numCols;
                uint16_t bias_length = bias_numRows * bias_numCols;

                // extract layer weights
                int16_t *kernel_array = &array[i];
                i += kernel_length;
                bias_array = &array[i];
                i += bias_length;

                // prepare output
                uint16_t output_numRows = kernel_numRows;
                uint16_t output_numCols = input->numCols;
                output->numRows = output_numRows;
                output->numCols = output_numCols;

                // initialize weight matrix
                matrix kernel = {kernel_array, kernel_numRows, kernel_numCols};
                matrix bias = {bias_array, bias_numRows, bias_numCols};

                // execute dense layer
                if (activation == RELU_ACTIVATION){
                    dense(output, input, &kernel, &bias, &fp_relu, FIXED_POINT_PRECISION);
                }
                else if (activation == SIGMOID_ACTIVATION){
                    dense(output, input, &kernel, &bias, &fp_sigmoid, FIXED_POINT_PRECISION);
                }
                else{
                    dense(output, input, &kernel, &bias, &fp_linear, FIXED_POINT_PRECISION);
                }
            }

            /* layer class 1 - LeakyReLU */
            else if (array[i] == LEAKY_RELU_LAYER){
                output->numRows = input->numRows;
                output->numCols = input->numCols;
                apply_leakyrelu(output, input, FIXED_POINT_PRECISION);
                i ++;
            }

            /* layer class 2 - Conv2D */
            else if (array[i] == CONV2D_LAYER){

                // extract and prepare layer parameters
                layer_class = array[i];
                activation = array[i+1];
                numFilters = array[i+2];
                numChannels = array[i+3];
                filter_numRows = array[i+4];
                filter_numCols = array[i+5];
                stride_numRows = array[i+6];
                stride_numCols = array[i+7];
                filters_length = array[i+8];
                padding = array[i+9];
                i += 10;

                // prepare output
                if (padding == 1){
                    output->numRows = input->numRows / stride_numRows;
                    if (input->numRows % stride_numRows > 0){
                        output->numRows ++;
                    }
                    output->numCols = input->numCols / stride_numCols;
                    if (input->numCols % stride_numRows > 0){
                        output->numCols ++;
                    }
                }
                else {
                    output->numRows = (input->numRows - filter_numRows + 1) / stride_numRows;
                    if ((input->numRows - filter_numRows + 1) % stride_numRows > 0){
                        output->numRows ++;
                    }
                    output->numCols = (input->numCols - filter_numCols + 1) / stride_numCols;
                    if ((input->numCols - filter_numCols + 1) % stride_numCols > 0){
                        output->numCols ++;
                    }
                }

                // extract and prepare weights
                int16_t *filters_array = array + i;
                matrix filters = {filters_array, filter_numRows, filter_numCols};
                i += filters_length;

                bias_array = array + i;
                i += numFilters;


                // execute conv2d layer
                if (activation == RELU_ACTIVATION){
                    conv2d(output, input, &filters, numFilters, numChannels, bias_array, &fp_relu, FIXED_POINT_PRECISION, stride_numRows, stride_numCols, padding);
                }
                else if (activation == SIGMOID_ACTIVATION){
                    conv2d(output, input, &filters, numFilters, numChannels, bias_array, &fp_sigmoid, FIXED_POINT_PRECISION, stride_numRows, stride_numCols, padding);
                }
                else{
                    conv2d(output, input, &filters, numFilters, numChannels, bias_array, &fp_linear, FIXED_POINT_PRECISION, stride_numRows, stride_numCols, padding);
                }

            }

            /* layer class 3 - MaxPooling2D */
            else if (array[i] == MAXPOOLING2D_LAYER){
                uint16_t pool_numRows = array[i+1];
                uint16_t pool_numCols = array[i+2];
                stride_numRows = array[i+3];
                stride_numCols = array[i+4];
                padding = array[i+5];
                i += 6;

                output->numRows = input->numRows / pool_numRows;
                output->numCols = input->numCols / pool_numCols;

                maxpooling_filters(output, input, numFilters, pool_numRows, pool_numCols);
            }

            /* layer class 4 - Conv2D Flatten */
            else if (array[i] == FLATTEN_LAYER){
                i += 1;
                output->numRows = input->numRows * input->numCols * numFilters;
                output->numCols = LEA_RESERVED;
                flatten(output, input, numFilters);
                numFilters = 1;
            }
            /* SKIP FOR INFERENCE TIME IMPLEMENTATION - layer class 5 - Dropout Layer */
            else if (array[i] == DROPOUT_LAYER){
                i += 1;
                numFilters = 1;
            }

            /* copy output matrix and reference input to copied output */
            dma_load(MODEL_ARRAY_TEMP, output->data, output->numRows * output->numCols * numFilters);
            input->data = MODEL_ARRAY_TEMP;
            input->numRows = output->numRows;
            input->numCols = output->numCols;
        }
    }

    return output;
}
