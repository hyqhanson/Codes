#ifndef CLASS
#define CLASS

#include <iostream>
#include <vector>
#include <universal/number/posit/posit.hpp>

using namespace sw::universal;
using namespace std;

template <typename P>
class MultipleLinearRegression
{
private:
    int numFeatures; // Number of features
    vector<P> theta; // Parameters
    P learningRate;  // Learning rate for gradient descent

public:
    MultipleLinearRegression(int numFeatures, double learningRate);

    // Hypothesis function for multiple linear regression
    P hypothesis(const vector<P> &features) const;

    // Train the model using gradient descent
    void train(const vector<vector<P>> &X, const vector<P> &y, int numIterations);

    // Make predictions using the trained model
    P predict(const vector<P> &features) const;

    // Get the learned parameters (theta values)
    vector<P> getTheta() const;

    P Cost(const vector<vector<P>> &X, const vector<P> &y) const;
};

#endif
