#include <stdlib.h>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "../lib/matrix-addition.cuh"

static void EXPECT_MATRIX_EQ(Matrix actual, Matrix expectedMatrix) {
    for (std::size_t i = 0; i < actual.height; ++i) {
        for (std::size_t j = 0; j < actual.width; ++j) {
            size_t index = (i * actual.width) + j;
            EXPECT_EQ(actual.elements[index], expectedMatrix.elements[index]);
        };
    };

    EXPECT_EQ(actual.height, expectedMatrix.height);
    EXPECT_EQ(actual.width, expectedMatrix.width);
};

TEST(MatrixAddition, TwoSingleElementMatrices) {
    const std::size_t width = 1, height = 1;

    int *resultMatrixElements = (int *)malloc(width * height * sizeof(int));
    int matrixOneElements[] = { 12 };
    int matrixTwoElements[] = { 24 };

    Matrix resultMatrix = Matrix(height, width, resultMatrixElements);
    Matrix matrixOne = Matrix(height, width, matrixOneElements);
    Matrix matrixTwo = Matrix(height, width, matrixTwoElements);

    addMatrices(resultMatrix, matrixOne, matrixTwo);

    int expectedMatrixElements[height * width] = { 36 };
    Matrix expectedMatrix = Matrix(height, width, expectedMatrixElements);

    EXPECT_MATRIX_EQ(resultMatrix, expectedMatrix);

    free(resultMatrixElements);
    cudaDeviceReset();
};

TEST(MatrixAddition, TwoSingleDimensionalMatrices) {
    const std::size_t width = 4, height = 1;

    int *resultMatrixElements = (int *)malloc(width * height * sizeof(int));
    int matrixOneElements[] = { 1, 2, 3, 4 };
    int matrixTwoElements[] = { 8, 16, 32, 64 };

    Matrix resultMatrix = Matrix(height, width, resultMatrixElements);
    Matrix matrixOne = Matrix(height, width, matrixOneElements);
    Matrix matrixTwo = Matrix(height, width, matrixTwoElements);

    addMatrices(resultMatrix, matrixOne, matrixTwo);

    int expectedMatrixElements[height * width] = { 9, 18, 35, 68 };
    Matrix expectedMatrix = Matrix(height, width, expectedMatrixElements);

    EXPECT_MATRIX_EQ(resultMatrix, expectedMatrix);

    free(resultMatrixElements);
    cudaDeviceReset();
}

TEST(MatrixAddition, TwoMultiDimensionalMatrices) {
    const std::size_t width = 2, height = 2;

    int *resultMatrixElements = (int *)malloc(width * height * sizeof(int));
    int matrixOneElements[] = { 1, 2, 3, 4 };
    int matrixTwoElements[] = { 8, 16, 32, 64 };

    Matrix resultMatrix = Matrix(height, width, resultMatrixElements);
    Matrix matrixOne = Matrix(height, width, matrixOneElements);
    Matrix matrixTwo = Matrix(height, width, matrixTwoElements);

    addMatrices(resultMatrix, matrixOne, matrixTwo);

    int expectedMatrixElements[height * width] = { 9, 18, 35, 68 };
    Matrix expectedMatrix = Matrix(height, width, expectedMatrixElements);

    EXPECT_MATRIX_EQ(resultMatrix, expectedMatrix);

    free(resultMatrixElements);
    cudaDeviceReset();
}

TEST(MatrixAddition, TwoLargeMultiDimensionalMatrices) {
    const std::size_t width = 2000, height = 3000;

    int *resultMatrixElements = (int *)malloc(width * height * sizeof(int));
    int *matrixOneElements = (int *)malloc(width * height * sizeof(int));
    int *matrixTwoElements = (int *)malloc(width * height * sizeof(int));
    int *expectedMatrixElements = (int *)malloc(width * height * sizeof(int));

    for(size_t i = 0; i < (width * height); ++i) {
        matrixOneElements[i] = i;
        matrixTwoElements[i] = 2 * i;
        expectedMatrixElements[i] = 3 * i;
    }

    Matrix resultMatrix = Matrix(height, width, resultMatrixElements);
    Matrix matrixOne = Matrix(height, width, matrixOneElements);
    Matrix matrixTwo = Matrix(height, width, matrixTwoElements);

    addMatrices(resultMatrix, matrixOne, matrixTwo);

    Matrix expectedMatrix = Matrix(height, width, expectedMatrixElements);

    EXPECT_MATRIX_EQ(resultMatrix, expectedMatrix);
    
    free(matrixOneElements);
    free(matrixTwoElements);
    free(expectedMatrixElements);
    cudaDeviceReset();
}
