/*
 * main.c
 * Include processing inputs, outputs and applying deep learning models
 */

#include "main.h"

//static dtype input[9] = {1024, 2048, 512, 0, 256, 1024, 2048, 0, 0};
//static matrix input_mat = {input, 3, 3};
//
//static dtype filter[4] = {1024, 512, 512, 0};
//static matrix filter_mat = {filter, 2, 2};
//
//static dtype result[4] = {0};
//static matrix result_mat = {result, 2, 2};

void main(void){

    /* stop watchdog timer */
     WDTCTL = WDTPW | WDTHOLD;

    /* initialize GPIO System */
    init_gpio();

    /* initialize the clock and baudrate */
    init_clock_system();

    /* GSC dataset consists of 32*32 spectrograms */
    inputFeatures.numRows = INPUT_NUM_ROWS;
    inputFeatures.numCols = INPUT_NUM_COLS;
    inputFeatures.data = input_buffer;

    /* output consists of 35 classes */
    outputLabels.numRows = OUTPUT_NUM_LABELS;
    outputLabels.numCols = LEA_RESERVED;   // one more column is reserved for LEA
    outputLabels.data = output_buffer;

    /* apply model saved in neural_network_parameters.h on input */
    apply_model(&outputLabels, &inputFeatures);
    label = argmax(&outputLabels);

    __no_operation();
}
