#include "handle-error.cuh"

__global__
void addVectorsKernel(int *resultVector, std::size_t length, int *vectorOne, int *vectorTwo) {
    for (std::size_t i = threadIdx.x; i < length; i += blockDim.x)
        resultVector[i] = vectorOne[i] + vectorTwo[i];
};

void addVectors(int *resultVector, std::size_t length, int *vectorOne, int *vectorTwo) {
    int *deviceVectorOne, *deviceVectorTwo, *deviceResultVector;
    size_t arraySizeInBytes = length * sizeof(int);

    checkCudaCall(cudaMalloc(&deviceVectorOne, arraySizeInBytes));
    checkCudaCall(cudaMalloc(&deviceVectorTwo, arraySizeInBytes));
    checkCudaCall(cudaMalloc(&deviceResultVector, arraySizeInBytes));

    checkCudaCall(cudaMemcpy(deviceVectorOne, vectorOne, arraySizeInBytes, cudaMemcpyHostToDevice));
    checkCudaCall(cudaMemcpy(deviceVectorTwo, vectorTwo, arraySizeInBytes, cudaMemcpyHostToDevice));

    std::size_t blockSize = 256;
    std::size_t numBlocks = (length + blockSize - 1) / blockSize;

    addVectorsKernel<<<numBlocks, blockSize>>>(deviceResultVector, length, deviceVectorOne, deviceVectorTwo);

    checkCudaCall(cudaFree(deviceVectorOne));
    checkCudaCall(cudaFree(deviceVectorTwo));

    checkCudaCall(cudaMemcpy(resultVector, deviceResultVector, arraySizeInBytes, cudaMemcpyDeviceToHost));
};
