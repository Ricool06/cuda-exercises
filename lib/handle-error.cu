#include <stdio.h>

static void checkCudaCall_f(cudaError_t cudaError, const char* file, int line) {
    if (cudaError != cudaSuccess) {
        fprintf(stderr, "ERROR: CUDA call failed in file: %s at line %d\n", file, line);
        exit(cudaError);
    }
};

void checkCudaCall(cudaError_t cudaError) {
    checkCudaCall_f(cudaError, __FILE__, __LINE__);
};
