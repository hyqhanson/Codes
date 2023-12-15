#include <iostream>
#include <vector>
#include <cmath>

#include "class.h"

using namespace std;

// Implementation of the constructor
template <typename P>
LogisticRegression<P>::LogisticRegression(const vector<vector<P>> &input_data, const vector<int> &input_labels,
                                          double learning_rate)
    : data(input_data), labels(input_labels), learningRate(learning_rate)
{
    // Initialize weights and bias
    weights.resize(data[0].size(), (P)rand() / RAND_MAX);
    bias = 0.0;
}

// Implementation of the sigmoid function
template <typename P>
P LogisticRegression<P>::sigmoid(P z) const
{
    return 1.0 / (1.0 + exp(-z));
}

// Implementation of the prediction function
template <typename P>
P LogisticRegression<P>::predict(const vector<P> &input) const
{
    P z = bias;

    for (size_t i = 0; i < input.size(); ++i)
    {
        z += weights[i] * input[i];
    }

    return sigmoid(z);
}


// Implementation of the training function
template <typename P>
void LogisticRegression<P>::train(int numIterations)
{
    size_t numFeatures = data[0].size();
    size_t numExamples = data.size();

    for (int iteration = 0; iteration < numIterations; ++iteration)
    {
        for (size_t i = 0; i < numExamples; ++i)
        {
            P y = labels[i];
            P prediction = predict(data[i]);
            P error = prediction - y;

            // Update weights and bias based on the error
            for (size_t j = 0; j < numFeatures; ++j)
            {
                weights[j] -= learningRate * error * data[i][j];
            }
            bias -= learningRate * error;
        }
       
    }
}

// Implementation of the displayParameters function
template <typename P>
void LogisticRegression<P>::displayParameters() const
{
    cout << "Learned Weights: ";
    for (P weight : weights)
    {
        cout << weight << " ";
    }
    cout << "\nLearned Bias: " << bias << endl;
}

template <typename P>
int LogisticRegression<P>::roundProbability(P probability, P threshold)
{
    return (probability >= threshold) ? 1 : 0;
}

/**
int main() {
    // Example usage for Logistic Regression
    vector<vector<P>> input_data = {{1, 2}, {2, 3}, {3, 4}, {4, 5}};
    vector<int> labels = {0, 0, 1, 1};  // Binary class labels (0 or 1)

    P learningRate = 0.01;
    int numIterations = 1000;

    LogisticRegression logistic_model(input_data, labels, learningRate);

    // Train the Logistic Regression
    logistic_model.train(numIterations);

    // Display the learned parameters
    logistic_model.displayParameters();

    // Test the Logistic Regression
    vector<P> test_input = {5, 6};
    P prediction = logistic_model.predict(test_input);

    cout << "Prediction: " << prediction << endl;

    return 0;
}
**/