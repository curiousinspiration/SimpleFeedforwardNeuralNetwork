/*
 * ReLU Layer Test
 *
 */

#include "neural/layers/relu_layer.h"

#include <gtest/gtest.h>

using namespace neural;
using namespace std;

// TEST(TestCaseName, IndividualTestName)
TEST(ReLULayerTest, TestForward)
{
        // Init input
    TMatrix input = {
        {-4.0, 3.0},
        {2.0, -1.0}
    };

    ReLULayer layer;

    // Relu layer should calculate max(0,x) for each value in the matrix
    TMatrix output = layer.Forward(input);
    EXPECT_EQ(2, output.size());
    EXPECT_EQ(2, output.at(0).size());

    /*
    (0,0) = max(0.0, -4.0) = 0
    (0,1) = max(0.0,  3.0) = 3.0
    (1,0) = max(0.0,  2.0) = 2.0
    (1,1) = max(0.0, -1.0) = 0
    */

    EXPECT_EQ(0.0, output.at(0).at(0));
    EXPECT_EQ(3.0, output.at(0).at(1));
    EXPECT_EQ(2.0, output.at(1).at(0));
    EXPECT_EQ(0.0, output.at(1).at(1));
}
