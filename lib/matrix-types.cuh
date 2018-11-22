#ifndef MATRIX_TYPES
#define MATRIX_TYPES

struct Matrix {
    size_t height;
    size_t width;
    int *elements;

    Matrix(size_t height,
        size_t width,
        int *elements):height(height), width(width), elements(elements) {}
};

#endif
