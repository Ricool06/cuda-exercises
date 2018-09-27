Requires:
* A CUDA enabled device (most modern NVIDIA GPUs)
* CUDA Toolkit

How to use:
1. `cd` into the folder of your choice
1. Run `nvcc main.cu -o main`
1. Run `nvprof ./main` to see performance profile of compiled binary
1. Run `./main` to just run compiled binary

Note:
Some folders will also contain a main.cpp file showing how the same program can be written in a serial, single-threaded manner.

Credit:
Some code essentially ripped from this tutorial: https://devblogs.nvidia.com/even-easier-introduction-cuda/
Programming Massively Parallel Processors by David B. Kirk and Wen-mei W. Hwu