/*
 * Relu Layer Implementation
 *
 */

#include "neural/layers/relu_layer.h"

#include <algorithm>

using namespace std;

namespace neural
{

ReLULayer::ReLULayer()
{

}

TMatrix ReLULayer::Forward(const TMatrix& a_input) const
{
    // initialize our return matrix with the input
    TMatrix l_ret = a_input;
 
    // walk over rows
    for (size_t i = 0; i < a_input.size(); ++i)
    {
        // walk over cols
        for (size_t j = 0; j < a_input.at(0).size(); ++j)
        {
            // max(0,x)
            l_ret.at(i).at(j) = std::max(0.0f, l_ret.at(i).at(j));
        }
    }
    return l_ret;
}

TMatrix ReLULayer::Backward(const TMatrix& a_origInput, const TMatrix& a_gradInput)
{
    TMatrix grad = a_gradInput;
    for (size_t i = 0; i < a_gradInput.size(); ++i)
    {
        // walk over cols
        for (size_t j = 0; j < a_origInput.at(0).size(); ++j)
        {
            if (a_origInput.at(i).at(j) < 0)
            {
                grad.at(i).at(j) = 0.0;
            }
        }
    }
    return grad;
}


} // namespace neural
