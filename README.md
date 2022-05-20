# Introduction

Sample script for loading and inferencing TensorFlow models in C.

Author: Muralikrishnan Gopalakrishnan Meena (Oak Ridge National Laboratory), https://sites.google.com/view/muraligm/

Contributors:
* Murali Gopalakrishnan Meena

# Description

These are sample scripts to load a TensorFlow model and run a forward step. See [TensorFlow for C](https://www.tensorflow.org/install/lang_c) for TF C API installation instructions.

The following examples are provided (will be updated accordingly):
1. [`example_tensorflow_HelloWorld.c`](example_tensorflow_HelloWorld.c): A sample Hello World implementation
2. [`example_tensorflow_NNmodel.c`](example_tensorflow_NNmodel.c): A simple model to replicate an activation function
    - Based on https://github.com/AmirulOm/tensorflow_capi_sample.git
3. [`example_tensorflow_tanh.c`](example_tensorflow_tanh.c): A simple feedforwad NN for regression
    - Model based on https://github.com/muralikrishnangm/tutorial-ai4science-fluidflow#example-1
4. [example_tensorflow_AICT.c](example_tensorflow_AICT.c): To load and perform inference on a CNN model. Used for the [2022 AAPM TrueCT Challenge](https://www.aapm.org/GrandChallenge/TrueCT/)

# Usage

First save any TensorFlow or Keras model in your python script as:
```
my_model.save('model_dir_name', save_format='tf')
```

1. Activate binaries using environment file corresponding to your machine
    ```
    source build/<machineName>_<cpu/gpu>.env
    ```
    Example: For running on CPU on Summit
    ```
    source build/summit_cpu.env
    ```
2. Compile: 
    Update `Makefile` to select which model to do inference (change `SRC` in the file)
    ```
    make
    ```
3. Run:
    ```
    ./example.exe models/<model-name>
    ```
    
* Sample output for `example_tensorflow_HelloWorld.c`:
  ```
  Hello from TensorFlow C library version 2.4.1
  ```


