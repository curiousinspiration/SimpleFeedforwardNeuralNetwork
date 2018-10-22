/*
 * Base Class for Layer
 *
 */

#pragma once

#include "neural/typedefs.h"

namespace neural
{

class Layer
{
public:
    virtual TMatrix Forward(const TMatrix& a_input) const = 0;
    virtual TMatrix Backward(const TMatrix& a_origInput, const TMatrix& a_gradInput) = 0;


};

} // namespace neural
