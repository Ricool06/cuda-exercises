__global__
void addVectorsKernel(int *resultVector, std::size_t length, int *vectorOne, int *vectorTwo) {
    for (std::size_t i = threadIdx.x; i < length; i += blockDim.x)
        resultVector[i] = vectorOne[i] + vectorTwo[i];
};

void addVectors(int *resultVector, std::size_t length, int *vectorOne, int *vectorTwo) {
    int *deviceVectorOne, *deviceVectorTwo, *deviceResultVector;
    size_t arraySizeInBytes = length * sizeof(int);

    cudaMalloc(&deviceVectorOne, arraySizeInBytes);
    cudaMalloc(&deviceVectorTwo, arraySizeInBytes);
    cudaMalloc(&deviceResultVector, arraySizeInBytes);

    cudaMemcpy(deviceVectorOne, vectorOne, arraySizeInBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceVectorTwo, vectorTwo, arraySizeInBytes, cudaMemcpyHostToDevice);

    std::size_t blockSize = 256;
    std::size_t numBlocks = (length + blockSize - 1) / blockSize;

    addVectorsKernel<<<numBlocks, blockSize>>>(deviceResultVector, length, deviceVectorOne, deviceVectorTwo);

    cudaFree(deviceVectorOne);
    cudaFree(deviceVectorTwo);

    cudaMemcpy(resultVector, deviceResultVector, arraySizeInBytes, cudaMemcpyDeviceToHost);
};
