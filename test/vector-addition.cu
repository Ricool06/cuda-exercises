#include <stdlib.h>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "../lib/vector-addition.cuh"

TEST(TestCaseName, TestName) {
    EXPECT_EQ(1, 1);
}

TEST(VectorAddition, TwoSingleElementArrays) {
    int *vectorOne, *vectorTwo, *resultVector;
    std::size_t length = 1;

    vectorOne = (int*) malloc(length * sizeof(int));
    vectorTwo = (int*) malloc(length * sizeof(int));
    resultVector = (int*) malloc(length * sizeof(int));

    vectorOne[0] = 12;
    vectorTwo[0] = 24;

    addVectors(resultVector, length, vectorOne, vectorTwo);

    static int expectedVector[] = { 36 };
    for (int i = 0; i < length; i++) {
        EXPECT_EQ(resultVector[i], expectedVector[i]);
    }

    free(vectorOne);
    free(vectorTwo);
    free(resultVector);
}

TEST(VectorAddition, TwoMultiElementArrays) {
    int *resultVector;
    std::size_t length = 4;

    resultVector = (int*) malloc(length * sizeof(int));

    static int vectorOne[] = { 2, 4, 6, 8 };
    static int vectorTwo[] = { 4, 8, 16, 32 };

    addVectors(resultVector, length, vectorOne, vectorTwo);

    static int expectedVector[] = { 6, 12, 22, 40 };
    for (int i = 0; i < length; i++) {
        EXPECT_EQ(resultVector[i], expectedVector[i]);
    }

    free(resultVector);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}