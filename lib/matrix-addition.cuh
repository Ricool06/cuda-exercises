#ifndef MATRIX_ADDITION
#define MATRIX_ADDITION

struct Matrix {
    size_t height;
    size_t width;
    int *elements;

    Matrix(size_t height,
        size_t width,
        int *elements):height(height), width(width), elements(elements) {}
};

void addMatrices(Matrix resultMatrix,
    Matrix matrixOne,
    Matrix matrixTwo);

#endif
