# Capuchin
## About This
This repository stores an automation tool designed to implement the inference of Python Tensorflow uDNN (micro deep neural network) models on TI MSP430FR5994 microcontrollers.

This project is made into two parts: a Python interface `encoder.py` and the microcontroller C implementation `uDNN-tf2msp-msp_impl`. The Python interface is built with a function that takes `tensorflow.keras.Model` objects as the argument to extract uDNN model configuration and generates a compiled header file `neural_network_parameters.h` that can be used to compile along with the microcontroller C implementation. The microcontroller C implementation extracts the model configuration array in the header file `neural_network_parameters.h`, builds a uDNN model on MSP430FR5994 by calling a sequence of uDNN layer implemetations, and feeds the sample dataset input into the uDNN model to compute the inference output.

## Getting Started
### Dependencies
Disclaim: the following dependencies are only author's recommended setup. This project was tested under the following dependencies and verified as correct. This project may work under other versions of dependencies but the correctness may not be guaranteed.

+ tensorflow=2.6.0
+ keras=2.6.0
+ fxpmath=0.4.0
+ numpy=1.19.5
+ TI Code Composer Studio (recommended to debug as this project only simulates input and output rather than connecting to actual sensors)
### Installation
`git clone https://github.com/leleonardzhang/Capuchin.git`
### Executing the Program
#### Using Python Interface
1. Copy `encoder.py` into the same directory of your Python uDNN implementation program.

    `cp uDNN-tf2msp/encoder.py {UDNN_PYTHON_IMPL_DIR}`
2. Import encoder package into your Python uDNN implementation program.

    `import encoder`
3. Call `encoder.export_model(MODEL_OBJECT)` in your program where the argument`MODEL_OBJECT` should be a trained `tensorflow.keras.Model` object.
4. Now a header file named `neural_network_parameters.h` should be written in the same directory.
#### Using C Implementation
1. Copy the header file `neural_network_parameters.h` into directory `uDNN-tf2msp/uDNN-tf2msp-msp_impl`.

    `cp {UDNN_PYTHON_IMPL_DIR}/neural_network_parameters.h {TF2MSP_PROJECT_DIR}/uDNN-tf2msp/uDNN-tf2msp-msp_impl`
2. Open `{TF2MSP_PROJECT_DIR}/uDNN-tf2msp/uDNN-tf2msp-msp_impl` as a TI CCS project.
3. Copy desired input into `input_buffer` array in `uDNN-tf2msp/uDNN-tf2msp-msp_impl/neural_network_parameters.h`.
4. Compile and Run the inference on MSP430.

## Supports
### Python Interface
For the reason that tensorflow is a complex framework for deep learning, the functionality of automation implementation on MSP430 is not comprehensive. Only frequent-used layers and functions in DNNs and CNNs are now supported. The following implementations are supported and verified as correct:
#### Tensorflow Models
+ Sequential
#### Tensorflow Layers (Parameters)
+ Dense (activation)
+ Conv2d (filters, channels, activation, strides, padding=`'valid' | 'same'`)
+ Maxpooling2d (pool_size)
+ Flatten
+ Dropout
+ LeakyReLU
#### Tensorflow Activation Functions
+ linear
+ ReLU
+ LeakyReLU
+ tanh
+ sigmoid
### C Implementation
Due to the hardware constraints of MSP430, the C implementation should meet the following known requirements:
+ `LENGTH_OF_INPUT + NUMBER_OF_PARAMETERS_IN_MODEL <= 68K` (FRAM persistent memory constraint)
+ `LEA_RAM_SIZE <= 1.8K` (LEA SRAM volatile memory constraint)
+ `LENGTH_OF_ANY_LAYER_OUTPUT <= 16384` (FRAM persistent memory constraint)
## Examples
[MNIST](examples/MNIST_uDNN.ipynb)
[CIFAR-10](examples/CIFAR_10_uDNN.ipynb)
## Contact
Le Zhang - Email: lezhang at unc.edu

Project Link: https://github.com/leleonardzhang/Capuchin

## Credits
This project is built on the framework of <a href="https://github.com/tejaskannan/budget-rnn">budget-RNN</a>.

T. Kannan and H. Hoffmann, "Budget RNNs: Multi-Capacity Neural Networks to Improve In-Sensor Inference Under Energy Budgets," 2021 IEEE 27th Real-Time and Embedded Technology and Applications Symposium (RTAS), 2021, pp. 143-156, doi: <a href="https://doi.org/10.1109/RTAS52030.2021.00020">10.1109/RTAS52030.2021.00020</a>.
