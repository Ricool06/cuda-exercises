#include <stdlib.h>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "../lib/matrix-multiplication.cuh"

static void EXPECT_MATRIX_EQ(Matrix actual, Matrix expectedMatrix) {
    for (std::size_t i = 0; i < expectedMatrix.height; ++i) {
        for (std::size_t j = 0; j < expectedMatrix.width; ++j) {
            size_t index = (i * expectedMatrix.width) + j;
            EXPECT_EQ(actual.elements[index], expectedMatrix.elements[index]) << "Error at\n\t row: " << i << "\n\t column: " << j;
        };
    };

    EXPECT_EQ(actual.height, expectedMatrix.height) << "Heights do not match";
    EXPECT_EQ(actual.width, expectedMatrix.width) << "Widths do not match";
};

TEST(MatrixMultiplication, TwoSingleElementMatrices) {
    const std::size_t width = 1, height = 1;

    int *resultMatrixElements = (int *)malloc(width * height * sizeof(int));
    int matrixOneElements[] = { 4 };
    int matrixTwoElements[] = { 8 };

    Matrix resultMatrix = Matrix(height, width, resultMatrixElements);
    Matrix matrixOne = Matrix(height, width, matrixOneElements);
    Matrix matrixTwo = Matrix(height, width, matrixTwoElements);

    multiplyMatrices(resultMatrix, matrixOne, matrixTwo);

    int expectedMatrixElements[height * width] = { 32 };
    Matrix expectedMatrix = Matrix(height, width, expectedMatrixElements);

    EXPECT_MATRIX_EQ(resultMatrix, expectedMatrix);

    free(resultMatrixElements);
    cudaDeviceReset();
};

TEST(MatrixMultiplication, TwoSingleRowAndColumnMatrices) {
    int *resultMatrixElements = (int *)malloc(sizeof(int));
    int matrixOneElements[] = { 4, 8, 12, 16 };
    int matrixTwoElements[] = { 1, 2, 3, 4 };

    Matrix resultMatrix = Matrix(1, 1, resultMatrixElements);
    Matrix matrixOne = Matrix(1, 4, matrixOneElements);
    Matrix matrixTwo = Matrix(4, 1, matrixTwoElements);

    multiplyMatrices(resultMatrix, matrixOne, matrixTwo);

    int expectedMatrixElements[] = { 120 };
    Matrix expectedMatrix = Matrix(1, 1, expectedMatrixElements);

    EXPECT_MATRIX_EQ(resultMatrix, expectedMatrix);

    free(resultMatrixElements);
    cudaDeviceReset();
};

TEST(MatrixMultiplication, TwoMultiRowAndColumnMatrices) {
    int *resultMatrixElements = (int *)malloc(4 * sizeof(int));
    int matrixOneElements[] =
      { 1, 2, 3, 4,
        1, 2, 3, 4 };
    int matrixTwoElements[] =
      { 1, 1,
        2, 2,
        3, 3,
        4, 4 };

    Matrix resultMatrix = Matrix(2, 2, resultMatrixElements);
    Matrix matrixOne = Matrix(2, 4, matrixOneElements);
    Matrix matrixTwo = Matrix(4, 2, matrixTwoElements);

    multiplyMatrices(resultMatrix, matrixOne, matrixTwo);

    int expectedMatrixElements[] =
      { 30, 30,
        30, 30 };
    Matrix expectedMatrix = Matrix(2, 2, expectedMatrixElements);

    EXPECT_MATRIX_EQ(resultMatrix, expectedMatrix);

    free(resultMatrixElements);
    cudaDeviceReset();
};

TEST(MatrixMultiplication, TwoMultiRowAndColumnMatricesOfDifferentAreas) {
    int *resultMatrixElements = (int *)malloc(6 * sizeof(int));
    int matrixOneElements[] = 
      { 1, 2, 3, 4,
        5, 6, 7, 8 };
    int matrixTwoElements[] =
      { 1, 5, 9,
        2, 6, 10,
        3, 7, 11,
        4, 8, 12 };

    Matrix resultMatrix = Matrix(2, 3, resultMatrixElements);
    Matrix matrixOne = Matrix(2, 4, matrixOneElements);
    Matrix matrixTwo = Matrix(4, 3, matrixTwoElements);

    multiplyMatrices(resultMatrix, matrixOne, matrixTwo);

    int expectedMatrixElements[] =
      { 30, 70, 110,
        70, 174, 278 };
    Matrix expectedMatrix = Matrix(2, 3, expectedMatrixElements);

    EXPECT_MATRIX_EQ(resultMatrix, expectedMatrix);

    free(resultMatrixElements);
    cudaDeviceReset();
};

TEST(MatrixMultiplication, TwoLargeMatrices) {
    Matrix matrixOne = Matrix(1000, 800, nullptr);
    Matrix matrixTwo = Matrix(800, 2000, nullptr);
    Matrix resultMatrix = Matrix(matrixOne.height, matrixTwo.width, nullptr);
    Matrix expectedMatrix = Matrix(matrixOne.height, matrixTwo.width, nullptr);

    matrixOne.elements = (int *)malloc(matrixOne.height * matrixOne.width * sizeof(int));
    matrixTwo.elements = (int *)malloc(matrixTwo.height * matrixTwo.width * sizeof(int));
    resultMatrix.elements = (int *)malloc(resultMatrix.height * resultMatrix.width * sizeof(int));
    expectedMatrix.elements = (int *)malloc(expectedMatrix.height * expectedMatrix.width * sizeof(int));

    for (size_t i = 0; i < (matrixOne.height * matrixOne.width); ++i) {
        matrixOne.elements[i] = i;
    }
    for (size_t i = 0; i < (matrixTwo.height * matrixTwo.width); ++i) {
        matrixTwo.elements[i] = i + 1;
    }

    for (size_t resultRow = 0; resultRow < resultMatrix.height; ++resultRow) {
        for (size_t resultCol = 0; resultCol < resultMatrix.width; ++resultCol) {
            size_t resultIndex = (resultRow * resultMatrix.width) + resultCol;
            expectedMatrix.elements[resultIndex] = 0;

            for (size_t k = 0; k < matrixOne.width; ++k) {
                expectedMatrix.elements[resultIndex] +=
                    matrixOne.elements[(resultRow * matrixOne.width) + k] *
                    matrixTwo.elements[(k * matrixTwo.width) + resultCol];
            }
        }
    }

    multiplyMatrices(resultMatrix, matrixOne, matrixTwo);

    EXPECT_MATRIX_EQ(resultMatrix, expectedMatrix);

    free(resultMatrix.elements);
    free(expectedMatrix.elements);
    free(matrixOne.elements);
    free(matrixTwo.elements);
    cudaDeviceReset();
};
