/*
 * Linear Layer Definition
 *
 */

#pragma once

#include "neural/layers/layer.h"

namespace neural
{

class LinearLayer : public Layer
{
public:
    LinearLayer(const TMatrix& a_weights, bool a_hasBias = true);
    virtual TMatrix Forward(const TMatrix& a_input) const override;
    virtual TMatrix Backward(const TMatrix& a_origInput, const TMatrix& a_gradInput) override;

    void ZeroWeightGrad();
    TMatrix GetAvgWeightGrad() const;
    void UpdateWeights(const TMatrix& a_gradient, float a_learningRate);
    TMatrix& GetMutableWeights();

private:
    bool m_hasBias;
    TMatrix m_weights;
    std::vector<TMatrix> m_weightGrads;
};

} // namespace
