/*
 * main.c
 * Include processing inputs, outputs and applying deep learning models
 */

#include "main.h"

void main(void){

    /* stop watchdog timer */
    WDTCTL = WDTPW | WDTHOLD;

    /* initialize GPIO System */
    init_gpio();

    /* initialize the clock and baudrate */
    init_clock_system();

    /* CIFAR-10 dataset consists of 32*32 RGB images */
    inputFeatures.numRows = INPUT_NUM_ROWS;
    inputFeatures.numCols = INPUT_NUM_COLS;
    inputFeatures.data = input_buffer;

    /* output consists of 10 classes */
    outputLabels.numRows = OUTPUT_NUM_LABELS;
    outputLabels.numCols = LEA_RESERVED;   // one more column is reserved for LEA
    outputLabels.data = output_buffer;

    /* apply model saved in neural_network_parameters.h on input */
    apply_model(&outputLabels, &inputFeatures);
    label = argmax(&outputLabels);

    __no_operation();
}
