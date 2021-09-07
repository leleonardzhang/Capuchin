#include "neural_network_parameters.h"
#include "math/matrix_ops.h"
#include "math/fixed_point_ops.h"
#include "math/matrix.h"
#include "utils/utils.h"
#include "layers/layers.h"

#ifndef DECODER_GUARD
#define DECODER_GUARD

#define DENSE_LAYER 0
#define LEAKY_RELU_LAYER 1
#define CONV2D_LAYER 2
#define MAXPOOLING2D_LAYER 3
#define FLATTEN_LAYER 4
#define DROPOUT_LAYER 5

#define LINEAR_ACTIVATION 0
#define RELU_ACTIVATION 2

matrix *apply_model(matrix *output, matrix *input);

#endif
