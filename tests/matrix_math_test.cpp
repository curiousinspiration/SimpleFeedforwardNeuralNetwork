/*
 * Matrix Math Test
 *
 */

#include "neural/math/matrix_math.h"

#include <gtest/gtest.h>

using namespace neural;
using namespace std;

// TEST(TestCaseName, IndividualTestName)
TEST(MatrixMathTest, TestMatMul)
{
    TMatrix lhs = {
        {4.0, 3.0},
        {2.0, 1.0}
    };

    TMatrix rhs = {
        {1.0, 2.0},
        {3.0, 4.0}
    };
    
    TMatrix result = MatrixMath::Multiply(lhs, rhs);
    EXPECT_EQ(2, result.size());
    EXPECT_EQ(2, result.at(0).size());

    /*
    (0,0) = 4*1 + 3*3 = 13
    (0,1) = 4*2 + 3*4 = 20
    (1,0) = 2*1 + 1*3 = 5
    (1,1) = 2*2 + 1*4 = 8
    */

    EXPECT_EQ(13.0, result.at(0).at(0));
    EXPECT_EQ(20.0, result.at(0).at(1));
    EXPECT_EQ(5.0,  result.at(1).at(0));
    EXPECT_EQ(8.0,  result.at(1).at(1));
}

TEST(MatrixMathTest, TestTranspose)
{
    TMatrix mat = {
        {5.0, 4.0, 3.0},
        {2.0, 1.0, 0.0}
    };

    
    TMatrix transpose = MatrixMath::Transpose(mat);
    EXPECT_EQ(3, transpose.size());
    EXPECT_EQ(2, transpose.at(0).size());

    /*
    5.0, 2.0,
    4.0, 1.0,
    3.0, 0.0
    */

    EXPECT_EQ(5.0, transpose.at(0).at(0));
    EXPECT_EQ(2.0, transpose.at(0).at(1));
    EXPECT_EQ(4.0, transpose.at(1).at(0));
    EXPECT_EQ(1.0, transpose.at(1).at(1));
    EXPECT_EQ(3.0, transpose.at(2).at(0));
    EXPECT_EQ(0.0, transpose.at(2).at(1));
}
