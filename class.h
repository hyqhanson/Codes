#ifndef CLASS_H
#define CLASS_H

#include <vector>
#include <universal/number/posit/posit.hpp>

template <typename P>
class MultipleLinearRegression
{
private:
    int numFeatures;      // Number of features
    std::vector<P> theta; // Parameters
    P learningRate;       // Learning rate for gradient descent

public:
    MultipleLinearRegression(int numFeatures, double learningRate, unsigned seed);

    // Hypothesis function for multiple linear regression
    P hypothesis(const std::vector<P> &features) const;

    // Train the model using gradient descent
    void train(const std::vector<std::vector<P>> &X, const std::vector<P> &y, int numIterations);

    // Make predictions using the trained model
    P predict(const std::vector<P> &features) const;

    // Get the learned parameters (theta values)
    std::vector<P> getTheta() const;

    P Cost(const std::vector<std::vector<P>> &X, const std::vector<P> &y) const;
};

template <typename P>
class LogisticRegression
{
private:
    std::vector<std::vector<P>> data; // Input data
    std::vector<int> labels;          // Class labels (0 or 1)
    std::vector<P> weights;           // Parameters (weights)
    P bias;                           // Bias term
    P learningRate;                   // Learning rate for optimization

public:
    LogisticRegression(const std::vector<std::vector<P>> &input_data, const std::vector<int> &input_labels,
                       double learning_rate);

    // Logistic function (sigmoid)
    P sigmoid(P z) const;

    // Logistic Regression prediction function
    P predict(const std::vector<P> &input) const;

    double crossEntropyLoss() const;

    // Train the Logistic Regression using gradient descent
    void train(int numIterations);

    // Display the learned parameters (weights and bias)
    void displayParameters() const;

    int roundProbability(P probability, P threshold);
};

inline double nearestPowerOfTwo(double num);

template <typename P>
void findMaxMin(std::vector<std::vector<P>> record_posit, P &Max, P &Min);

template <typename P>
void minMaxScaling(std::vector<std::vector<P>> record_posit, P Max, P Min, std::vector<std::vector<P>> &new_record);

#endif
