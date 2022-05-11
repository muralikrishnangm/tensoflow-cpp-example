# tensoflow-cpp-example

Sample script for loading and inferencing TensorFlow models in C++.

Author: Muralikrishnan Gopalakrishnan Meena (Oak Ridge National Laboratory), https://sites.google.com/view/muraligm/

Contributors:
* Murali Gopalakrishnan Meena

# Description

These are sample scripts to load a TensorFlow model and run a forward step. See [TensorFlow for C](https://www.tensorflow.org/install/lang_c) for TF C++ API installation instructions.

The following examples are provided (will be updated accordingly):
1. A sample Hello World implementation [`example_tensorflowNN_cpu.cpp`](example_tensorflowNN_cpu.cpp)

# Usage

1. Activate binaries using environment file corresponding to your machine
    ```
    source build/<machineName>_<cpu/gpu>.env
    ```
    Example: For running on CPU on Summit
    ```
    source summit_cpu.env
    ```
2. Compile: 
    ```
    make
    ```
3. Run:
    ```
    ./example.exe
    ```
    
* Sample output for the GPU usage (`example_tensorflowNN_cpu.cpp`)
  ```
  Hello from TensorFlow C library version 2.4.1
  ```


