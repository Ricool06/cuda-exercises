#include "matrix-addition.cuh"
#include "handle-error.cuh"

__global__
void addMatricesKernel(Matrix resultMatrix, Matrix matrixOne, Matrix matrixTwo) {
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < resultMatrix.height * resultMatrix.width) {
        resultMatrix.elements[id] = matrixOne.elements[id] + matrixTwo.elements[id];
    }
}

void addMatrices(Matrix resultMatrix, Matrix matrixOne, Matrix matrixTwo) {
    size_t resultMatrixSize = resultMatrix.height * resultMatrix.width;
    size_t resultMatrixSizeInBytes = resultMatrixSize * sizeof(int);

    Matrix dResultMatrix = Matrix(resultMatrix.height, resultMatrix.width, nullptr);
    Matrix dMatrixOne = Matrix(matrixOne.height, matrixOne.width, nullptr);
    Matrix dMatrixTwo = Matrix(matrixTwo.height, matrixTwo.width, nullptr);

    checkCudaCall(cudaMalloc(&dResultMatrix.elements, resultMatrixSizeInBytes));
    checkCudaCall(cudaMalloc(&dMatrixOne.elements, resultMatrixSizeInBytes));
    checkCudaCall(cudaMalloc(&dMatrixTwo.elements, resultMatrixSizeInBytes));

    checkCudaCall(cudaMemcpy(dMatrixOne.elements, matrixOne.elements, resultMatrixSizeInBytes, cudaMemcpyHostToDevice));
    checkCudaCall(cudaMemcpy(dMatrixTwo.elements, matrixTwo.elements, resultMatrixSizeInBytes, cudaMemcpyHostToDevice));

    size_t blockSize = 1024;
    size_t numBlocks = (resultMatrixSize + blockSize - 1) / blockSize;

    addMatricesKernel<<<numBlocks, blockSize>>>(dResultMatrix, dMatrixOne, dMatrixTwo);

    checkCudaCall(cudaFree(dMatrixOne.elements));
    checkCudaCall(cudaFree(dMatrixTwo.elements));

    checkCudaCall(cudaMemcpy(resultMatrix.elements, dResultMatrix.elements, resultMatrixSizeInBytes, cudaMemcpyDeviceToHost));

    checkCudaCall(cudaFree(dResultMatrix.elements));
}
