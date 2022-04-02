import tensorflow as tf
from tensorflow.keras import datasets, layers, models, Input, activations
from tensorflow.keras import initializers
from tensorflow import keras
from fxpmath import Fxp
import numpy as np
import math

def encode_model(model) -> int:
    if isinstance(model, models.Sequential): return 0
    return -1

def encode_layer(layer) -> int:
    if isinstance(layer, layers.Dense): return 0
    if isinstance(layer, layers.LeakyReLU): return 1
    if isinstance(layer, layers.Conv2D): return 2
    if isinstance(layer, layers.MaxPooling2D): return 3
    if isinstance(layer, layers.Flatten): return 4
    if isinstance(layer, layers.Dropout): return 5
    return -1

def encode_activation(layer) -> int:
    activation_name = layer.activation.__name__
    map_activation = {"linear":0, "sigmoid":1, "relu":2, "tanh":3, "softmax":4}
    return map_activation[activation_name]

def encode_dense_data(layer) -> list:
    data = []
    data.append(encode_activation(layer))
    fixed_point_arr = []
    weight = layer.get_weights()[0]
    if len(weight.shape) == 1:
        data.append(weight.shape[0])
        data.append(1)
    else:
        weight = weight.T
        data.append(weight.shape[0])
        data.append(weight.shape[1])
    
    bias = layer.get_weights()[1]
    if len(bias.shape) == 1:
        data.append(bias.shape[0])
        data.append(1)
    else:
        bias = bias.transpose(1, 0)
        data.append(bias.shape[0])
        data.append(bias.shape[1])
    
    fixed_point_weight = Fxp(weight, signed = True, n_word = 16, n_frac = 10).val.flatten().tolist()
    fixed_point_bias = Fxp(bias, signed = True, n_word = 16, n_frac = 10).val.flatten().tolist()
    data = data + fixed_point_weight + fixed_point_bias
    
    return data

def encode_padding(layer) -> int:
    if layer.padding == "valid":
        return 0
    if layer.padding == "same":
        return 1
    if layer.padding == "full":
        return 2
    return -1
    

def encode_conv2d_data(layer) -> list:
    data = []
    data.append(encode_activation(layer))       # activation function
    data.append(layer.filters)                  # num of filters
    if layer.data_format == "channels_last":    # num of channels
        data.append(layer.input_shape[3])
    else:
        data.append(layer.input_shape[0])
    data.append(layer.kernel_size[0])           # filter rows
    data.append(layer.kernel_size[1])           # filter cols
    
    data.append(layer.strides[0])               # stride rows
    data.append(layer.strides[1])               # stride cols
    data.append(layer.kernel_size[0] * layer.kernel_size[1] * layer.input_shape[3] * layer.filters)

    data.append(encode_padding(layer))          # padding

    weight = layer.get_weights()[0]
    weight = weight.transpose(3, 2, 0, 1)
    fixed_point_weight = Fxp(weight, signed = True, n_word = 16, n_frac = 10).val.flatten().tolist()

    bias = layer.get_weights()[1]
    fixed_point_bias = Fxp(bias, signed = True, n_word = 16, n_frac = 10).val.flatten().tolist()

    data = data + fixed_point_weight + fixed_point_bias

    return data


def encode_maxpooling2d_data(layer) -> list:
    data = []
    data.append(layer.pool_size[0])
    data.append(layer.pool_size[1])
    data.append(layer.strides[0])
    data.append(layer.strides[1])
    data.append(encode_padding(layer))          # padding
    return data


def encode(model) -> list:
    encode_list = []
    if encode_model(model) == -1:
        return encode_list
    encode_list.append(encode_model(model))

    if encode_model(model) == 0:
        for layer in model.layers:
            if encode_layer(layer) == -1:
                return encode_list
            if encode_layer(layer) == 0:
                encode_list.append(encode_layer(layer))
                encode_list = encode_list + encode_dense_data(layer)
            elif encode_layer(layer) == 1:
                encode_list.append(encode_layer(layer))
            elif encode_layer(layer) == 2:
                encode_list.append(encode_layer(layer))
                encode_list = encode_list + encode_conv2d_data(layer)
            elif encode_layer(layer) == 3:
                encode_list.append(encode_layer(layer))
                encode_list = encode_list + encode_maxpooling2d_data(layer)
            elif encode_layer(layer) == 4:
                encode_list.append(encode_layer(layer))
            elif encode_layer(layer) == 5:
                encode_list.append(encode_layer(layer))
    return encode_list


def export_model(model):
    count = 0
    output_str = ""
    for element in encode(model):
        output_str = output_str + f"{element},"
        count += 1
    left_bracket = '{'
    right_bracket = '}'
    output_str = f"""#include <stdint.h>
#include "math/matrix.h"

#ifndef NEURAL_NETWORK_PARAMS_GUARD
#define NEURAL_NETWORK_PARAMS_GUARD

#define FIXED_POINT_PRECISION 10
#define NUM_OUTPUTS 1
#define IS_MSP
#define LEA_RAM_LENGTH 1892
#define LEA_RESERVED 2

#define MODEL_ARRAY_LENGTH {len(encode(model))}
#define MODEL_ARRAY_OUTPUT_LENGTH 16384
#define MODEL_ARRAY_TEMP_LENGTH 16384
#define PADDING_BUFFER_LENGTH 2048
#define FILTER_BUFFER_LENGTH 1024
#define INPUT_NUM_ROWS {model.layers[0].input_shape[1]}
#define INPUT_NUM_COLS {model.layers[0].input_shape[2]}
#define INPUT_NUM_CHANNELS {model.layers[0].input_shape[3]}
#define OUTPUT_NUM_LABELS {model.layers[-1].get_weights()[0].shape[1]}

#define INPUT_LENGTH (INPUT_NUM_ROWS*INPUT_NUM_COLS*INPUT_NUM_CHANNELS)
#define OUTPUT_LENGTH (OUTPUT_NUM_LABELS*LEA_RESERVED)

#pragma LOCATION(MODEL_ARRAY, 0x18000)
#pragma PERSISTENT(MODEL_ARRAY)
static dtype MODEL_ARRAY[MODEL_ARRAY_LENGTH] = {left_bracket} {output_str[:-1]} {right_bracket};

/* INPUT HERE */
#pragma PERSISTENT(input_buffer)
static dtype input_buffer[INPUT_LENGTH] = {left_bracket} {right_bracket};
#pragma PERSISTENT(output_buffer)
static dtype output_buffer[OUTPUT_LENGTH] = {left_bracket}0{right_bracket};
static dtype label;
static matrix inputFeatures, outputLabels;

#endif
"""

    f = open("neural_network_parameters.h", "w")
    f.write(output_str)
    f.close()
