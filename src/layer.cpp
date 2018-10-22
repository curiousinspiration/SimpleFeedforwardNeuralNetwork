/*
 * Layer Implementation
 *
 */

#include "neural/layers/layer.h"

using namespace std;

namespace neural
{

void Layer::ZeroGrad()
{
    m_grads.clear();
}

TMatrix Layer::GetAvgGrad() const
{
    TMatrix average;
    // Init with zeros
    for (size_t i = 0; i < m_grads.at(0).size(); ++i)
    {
        average.push_back(vector<float>());
        for (size_t j = 0; j < m_grads.at(0).at(i).size(); ++j)
        {
            average.at(i).push_back(0.0);
        }
    }

    // Sum up
    for (const TMatrix& grad : m_grads)
    {
        for (size_t i = 0; i < grad.size(); ++i)
        {
            for (size_t j = 0; j < grad.at(i).size(); ++j)
            {
                average.at(i).at(j) += grad.at(i).at(j);
            }
        }
    }

    // Average
    float numGrads = (float)m_grads.size();
    for (size_t i = 0; i < m_grads.at(0).size(); ++i)
    {
        for (size_t j = 0; j < m_grads.at(0).at(i).size(); ++j)
        {
            average.at(i).at(j) /= numGrads;
        }
    }

    return average;
}

} // namespace neural
