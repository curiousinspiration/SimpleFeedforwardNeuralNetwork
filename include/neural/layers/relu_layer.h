/*
 * ReLU Layer Definition
 *
 */

#pragma once

#include "neural/layers/layer.h"

namespace neural
{

class ReLULayer : public Layer
{
public:
    ReLULayer();
    virtual TMatrix Forward(const TMatrix& a_input) const override;
    virtual TMatrix Backward(const TMatrix& a_origInput, const TMatrix& a_gradInput) override;

private:

};

} // namespace neural
