
#include "neural/layers/linear_layer.h"
#include "neural/layers/relu_layer.h"
#include "neural/loss/squared_error_loss.h"

#include <glog/logging.h>

using namespace neural;
using namespace std;

float CalcAverage(const vector<float>& vals)
{
    float sum = 0.0;
    for (size_t i = 0; i < vals.size(); ++i)
    {
        sum += vals.at(i);
    }
    return sum / ((float)vals.size());
}

int main(int argc, char const *argv[])
{
    // Define dataset
    vector<TMatrix> inputs = {
        TMatrix({{1.0, 1.0}}),
        TMatrix({{1.0, 0.0}}),
        TMatrix({{0.0, 1.0}}),
    };

    vector<float> outputs = {
        1.0,
        0.0,
        -1.0
    };

    // Define model

    // first linear layer is 2x2
    // 2 inputs, 2 outputs
    // Therefore needs 4 weights
    LinearLayer firstLinearLayer({
        {-0.5, 1.2},
        {0.6, -0.8}
    });

    // Non-linear activation
    ReLULayer activationLayer;
    
    // second linear layer is 2x1
    // 2 inputs, 1 output
    // Therefore needs 2 weights
    LinearLayer secondLinearLayer({
        {-0.5},
        {1.2}
    });

    // Error function
    SquaredErrorLoss loss;

    // Training loop
    float learningRate = 0.01;
    size_t numEpochs = 1000;
    for (size_t i = 0; i < numEpochs; ++i)
    {
        LOG(INFO) << "--EPOCH (" << i << ")--" << endl;
        vector<float> errorAcc;
        for (size_t j = 0; j < inputs.size(); ++j)
        {
            LOG(INFO) << "--ITER (" << i << "," << j << ")--" << endl;
            // Get training example
            TMatrix input = inputs.at(j);
            float targetOutput = outputs.at(j);

            // Forward pass
            TMatrix output0 = firstLinearLayer.Forward(input);
            TMatrix output1 = activationLayer.Forward(output0);
            TMatrix y_pred = secondLinearLayer.Forward(output1);

            float yPredVal = y_pred.at(0).at(0);

            // Calc Error
            float error = loss.Forward(yPredVal, targetOutput);
            errorAcc.push_back(error);

            // Print results
            LOG(INFO) << "Got prediction: " << yPredVal << " for target " << targetOutput << endl;
            LOG(INFO) << "Calculated error: " << error << endl;

            float errorGrad = loss.Backward(yPredVal, targetOutput);
            LOG(INFO) << "Got error grad: " << errorGrad << endl;

            LOG(INFO) << "secondLinearLayer.Backward" << endl;
            TMatrix y_predGrad = secondLinearLayer.Backward(output1, {{errorGrad}});

            LOG(INFO) << "activationLayer.Backward" << endl;
            TMatrix grad1 = activationLayer.Backward(output0, y_predGrad);

            LOG(INFO) << "firstLinearLayer.Backward" << endl;
            TMatrix grad0 = firstLinearLayer.Backward(input, grad1);
        }

        // Compute average error
        float avgError = CalcAverage(errorAcc);
        LOG(INFO) << "avgError = " << avgError << endl;

        LOG(INFO) << "UPDATE WEIGHTS!!!" << endl;
        // Gradient Descent
        secondLinearLayer.UpdateWeights(secondLinearLayer.GetAvgGrad(), learningRate);
        firstLinearLayer.UpdateWeights(firstLinearLayer.GetAvgGrad(), learningRate);

        // Clear gradients for next loop
        loss.ZeroGrad();
        secondLinearLayer.ZeroGrad();
        activationLayer.ZeroGrad();
        firstLinearLayer.ZeroGrad();
    }

    return 0;
}