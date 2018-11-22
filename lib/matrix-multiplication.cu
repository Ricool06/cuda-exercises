#include "matrix-multiplication.cuh"
#include "handle-error.cuh"

__global__
void multiplyMatricesKernel(Matrix resultMatrix, Matrix matrixOne, Matrix matrixTwo) {
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t column = blockIdx.x * blockDim.x + threadIdx.x;
    size_t resultIndex = (row * resultMatrix.width) + column;

    if (row < matrixOne.height && column < matrixTwo.width) {
        
        int result = 0;
        for(size_t i = 0; i < matrixOne.width; i++) {
            result += (matrixOne.elements[(row * matrixOne.width) + i]) * (matrixTwo.elements[(i * matrixTwo.width) + column]);
        }

        resultMatrix.elements[resultIndex] = result;
    }
}

void multiplyMatrices(Matrix resultMatrix, Matrix matrixOne, Matrix matrixTwo) {
    static const size_t tileSize = 32;

    size_t resultMatrixSizeInBytes = resultMatrix.height * resultMatrix.width * sizeof(int);
    size_t matrixOneSizeInBytes = matrixOne.height * matrixOne.width * sizeof(int);
    size_t matrixTwoSizeInBytes = matrixTwo.height * matrixTwo.width * sizeof(int);

    Matrix dResultMatrix(matrixOne.height, matrixTwo.width, nullptr);
    Matrix dMatrixOne(matrixOne.height, matrixOne.width, nullptr);
    Matrix dMatrixTwo(matrixTwo.height, matrixTwo.width, nullptr);

    checkCudaCall(cudaMalloc(&dResultMatrix.elements, resultMatrixSizeInBytes));
    checkCudaCall(cudaMalloc(&dMatrixOne.elements, matrixOneSizeInBytes));
    checkCudaCall(cudaMalloc(&dMatrixTwo.elements, matrixTwoSizeInBytes));

    checkCudaCall(cudaMemcpy(dMatrixOne.elements, matrixOne.elements, matrixOneSizeInBytes, cudaMemcpyHostToDevice));
    checkCudaCall(cudaMemcpy(dMatrixTwo.elements, matrixTwo.elements, matrixTwoSizeInBytes, cudaMemcpyHostToDevice));

    size_t xBlocksCount = (resultMatrix.width + tileSize - 1) / tileSize;
    size_t yBlocksCount = (resultMatrix.height + tileSize - 1) / tileSize;
    dim3 gridDimensions(xBlocksCount, yBlocksCount);
    dim3 blockDimensions(tileSize, tileSize);

    multiplyMatricesKernel<<<gridDimensions, blockDimensions>>>(dResultMatrix, dMatrixOne, dMatrixTwo);
    checkCudaCall(cudaDeviceSynchronize());

    checkCudaCall(cudaMemcpy(resultMatrix.elements, dResultMatrix.elements, resultMatrixSizeInBytes, cudaMemcpyDeviceToHost));

    checkCudaCall(cudaFree(dMatrixOne.elements));
    checkCudaCall(cudaFree(dMatrixTwo.elements));
    checkCudaCall(cudaFree(dResultMatrix.elements));
}