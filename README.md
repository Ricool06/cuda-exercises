Requires:
* CUDA Toolkit

How to use:
1. `cd` into the folder of your choice
1. Run `nvcc main.cu -o main`
1. Run `nvprof ./main` to see performance profile of compiled binary
1. Run `./main` to just run compiled binary

Note:
Some folders will also contain a main.cpp file showing how the same program can be written in a serial, single-threaded manner.