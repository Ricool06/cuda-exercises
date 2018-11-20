#include <stdlib.h>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "../lib/vector-addition.cuh"

TEST(TestCaseName, TestName) {
    EXPECT_EQ(1, 1);
}

TEST(VectorAddition, TwoSingleElementArrays) {
    std::size_t length = 1;
    int *resultVector = (int*) malloc(length * sizeof(int));

    static int vectorOne[] = { 12 };
    static int vectorTwo[] = { 24 };

    addVectors(resultVector, length, vectorOne, vectorTwo);

    static int expectedVector[] = { 36 };
    for (int i = 0; i < length; i++) {
        EXPECT_EQ(resultVector[i], expectedVector[i]);
    }

    free(resultVector);
}

TEST(VectorAddition, TwoMultiElementArrays) {
    std::size_t length = 4;
    int *resultVector = (int*) malloc(length * sizeof(int));

    static int vectorOne[] = { 2, 4, 6, 8 };
    static int vectorTwo[] = { 4, 8, 16, 32 };

    addVectors(resultVector, length, vectorOne, vectorTwo);

    static int expectedVector[] = { 6, 12, 22, 40 };
    for (int i = 0; i < length; i++) {
        EXPECT_EQ(resultVector[i], expectedVector[i]);
    }

    free(resultVector);
}

TEST(VectorAddition, TwoVeryLargeArrays) {
    std::size_t length = 6000000;
    int *resultVector = (int*) malloc(length * sizeof(int));
    int *expectedVector = (int*) malloc(length * sizeof(int));
    int *vectorOne = (int*) malloc(length * sizeof(int));
    int *vectorTwo = (int*) malloc(length * sizeof(int));

    for (std::size_t i = 0; i < length; i++) {
        vectorOne[i] = i;
        vectorTwo[i] = i;
        expectedVector[i] = 2 * i;
    }

    addVectors(resultVector, length, vectorOne, vectorTwo);

    for (int i = 0; i < length; i++) {
        EXPECT_EQ(resultVector[i], expectedVector[i]);
    }

    free(resultVector);
    free(expectedVector);
    free(vectorOne);
    free(vectorTwo);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}