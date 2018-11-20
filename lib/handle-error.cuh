#ifndef HANDLE_ERROR
#define HANDLE_ERROR

#include <stdio.h>

#define checkCudaCall(cudaError) {checkCudaCall_f(cudaError, __FILE__, __LINE__);}
inline void checkCudaCall_f(cudaError_t cudaError, const char* file, int line) {
    if (cudaError != cudaSuccess) {
        fprintf(stderr, "ERROR: CUDA call failed in file: %s at line %d\n", file, line);
        exit(cudaError);
    }
};

#endif