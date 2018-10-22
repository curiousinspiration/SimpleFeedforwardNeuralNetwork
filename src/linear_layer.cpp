/*
 * Linear Layer Implementation
 *
 */

#include "neural/layers/linear_layer.h"
#include "neural/math/matrix_math.h"

#include <glog/logging.h>

#include <sstream>

using namespace std;

namespace neural
{

LinearLayer::LinearLayer(const TMatrix& a_weights, bool a_hasBias)
    : m_hasBias(a_hasBias)
    , m_weights(a_weights)
{
    // if there is a bias, add an extra row to the weights
    if (m_hasBias)
    {
        vector<float> l_biasRow;
        for (size_t i = 0; i < a_weights.at(0).size(); ++i)
        {
            l_biasRow.push_back(1.0);
        }
        m_weights.push_back(l_biasRow);
    }
}

TMatrix LinearLayer::Forward(const TMatrix& a_input) const
{
    // Make a local copy so we can add bias if needed
    TMatrix l_input = a_input;

    if (m_hasBias)
    {
        // add an extra column of 1s
        for (size_t i = 0; i < l_input.size(); ++i)
        {
            l_input.at(i).push_back(1.0);
        }
    }

    return MatrixMath::Multiply(l_input, m_weights);
}

TMatrix LinearLayer::Backward(const TMatrix& a_origInput, const TMatrix& a_gradInput)
{
    LOG(INFO) << "LinearLayer::ComputeGrad a_origInput: " << a_origInput.size() << "x" << a_origInput.at(0).size() << endl;
    LOG(INFO) << "LinearLayer::ComputeGrad a_gradInput: " << a_gradInput.size() << "x" << a_gradInput.at(0).size() << endl;
    LOG(INFO) << "LinearLayer::ComputeGrad m_weights: " << m_weights.size() << "x" << m_weights.at(0).size() << endl;
    
    // orig input might have had bias
    TMatrix l_input = a_origInput;
    if (m_hasBias)
    {
        // add an extra column of 1s
        for (size_t i = 0; i < l_input.size(); ++i)
        {
            l_input.at(i).push_back(1.0);
        }
    }

    // Gradient wrt weights will be ........
    TMatrix gradWrtWeights = MatrixMath::Multiply(MatrixMath::Transpose(l_input), a_gradInput);
    LOG(INFO) << "LinearLayer::ComputeGrad gradWrtWeights: " << gradWrtWeights.size() << "x" << gradWrtWeights.at(0).size() << endl;
    m_grads.push_back(gradWrtWeights);

    // Gradient with respect to output will be (grad from last layer) * weights
    // Weights is the gradient with the respect to output for this layer
    // and we multipy because of the chain rule
    TMatrix gradWrtOutput = MatrixMath::Multiply(a_gradInput, MatrixMath::Transpose(m_weights));
    LOG(INFO) << "LinearLayer::ComputeGrad gradWrtOutput: " << gradWrtOutput.size() << "x" << gradWrtOutput.at(0).size() << endl;

    if (m_hasBias)
    {
        // delete last column because we don't need to pass back bias info
        for (size_t i = 0; i < gradWrtOutput.size(); ++i)
        {
            gradWrtOutput.at(i).pop_back();
        }
    }

    return gradWrtOutput;
}

void LinearLayer::UpdateWeights(const TMatrix& a_gradient, float a_learningRate)
{
    LOG(INFO) << "LinearLayer::UpdateWeights a_gradient: " << a_gradient.size() << "x" << a_gradient.at(0).size() << endl;
    LOG(INFO) << "LinearLayer::UpdateWeights m_weights: " << m_weights.size() << "x" << m_weights.at(0).size() << endl;
    for (size_t i = 0; i < m_weights.size(); ++i)
    {
        for (size_t j = 0; j < m_weights.at(i).size(); ++j)
        {
            m_weights.at(i).at(j) -= a_learningRate * a_gradient.at(i).at(j);
        }
    }
}

TMatrix& LinearLayer::GetMutableWeights()
{
    return m_weights;
}

} // namespace neural
