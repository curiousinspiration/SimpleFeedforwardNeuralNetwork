/*
 * Linear Layer Test
 *
 */

#include "neural/layers/linear_layer.h"

#include <gtest/gtest.h>

using namespace neural;
using namespace std;

// TEST(TestCaseName, IndividualTestName)
TEST(LinearLayerTest, TestForward)
{
    // Init input
    TMatrix input = {
        {4.0, 3.0},
        {2.0, 1.0}
    };

    // Init linear layer with weights
    TMatrix weights = {
        {1.0, 2.0},
        {3.0, 4.0}
    };
    bool hasBias = false;
    LinearLayer layer(weights, hasBias);

    // Linear layer should calculate linear transform of inputs*weights
    // y = xW
    TMatrix output = layer.Forward(input);
    EXPECT_EQ(2, output.size());
    EXPECT_EQ(2, output.at(0).size());

    /*
    (0,0) = 4*1 + 3*3 = 13
    (0,1) = 4*2 + 3*4 = 20
    (1,0) = 2*1 + 1*3 = 5
    (1,1) = 2*2 + 1*4 = 8
    */

    EXPECT_EQ(13.0, output.at(0).at(0));
    EXPECT_EQ(20.0, output.at(0).at(1));
    EXPECT_EQ(5.0,  output.at(1).at(0));
    EXPECT_EQ(8.0,  output.at(1).at(1));
}

TEST(LinearLayerTest, TestForwardWithBias)
{
        // Init input
    TMatrix input = {
        {4.0, 3.0},
        {2.0, 1.0}
    };

    // Init linear layer with weights
    TMatrix weights = {
        {1.0, 2.0},
        {3.0, 4.0}
    };
    bool hasBias = true;
    LinearLayer layer(weights, hasBias);

    // Linear layer should calculate linear transform of inputs*weights+b
    /*
    Our new matrices with the bias will look like

    input 2x3
    4.0, 3.0, 1.0,
    2.0, 1.0, 1.0

    weights 3x2
    1.0, 2.0,
    3.0, 4.0,
    1.0, 1.0
    */

    TMatrix output = layer.Forward(input);
    EXPECT_EQ(2, output.size());
    EXPECT_EQ(2, output.at(0).size());

    /*
    (0,0) = 4*1 + 3*3 + 1*1 = 14
    (0,1) = 4*2 + 3*4 + 1*1 = 21
    (1,0) = 2*1 + 1*3 + 1*1 = 6
    (1,1) = 2*2 + 1*4 + 1*1 = 9
    */

    EXPECT_EQ(14.0, output.at(0).at(0));
    EXPECT_EQ(21.0, output.at(0).at(1));
    EXPECT_EQ(6.0,  output.at(1).at(0));
    EXPECT_EQ(9.0,  output.at(1).at(1));
}
