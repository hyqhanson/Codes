#include <iostream>
#include <vector>

#include "class.h"

using namespace std;

template <typename P>
MultipleLinearRegression<P>::MultipleLinearRegression(int numFeatures, double learningRate)
    : numFeatures(numFeatures), learningRate(learningRate)
{
    // Initialize theta with zeros
    theta.resize(numFeatures + 1, 1.0);
}

// Hypothesis function for multiple linear regression
template <typename P>
P MultipleLinearRegression<P>:: hypothesis(const vector<P> &features) const
{
    P result = theta[0]; // Initialize with the bias term

    // Multiply each feature by its corresponding theta and sum them up
    for (int i = 0; i < numFeatures; ++i)
    {
        result += theta[i+1] * features[i];
    }
   
    return result;
}

// Train the model using gradient descent
template <typename P>
void MultipleLinearRegression<P>::train(const vector<vector<P>> &X, const vector<P> &y, int numIterations)
{
    int m = X.size(); // Number of training examples

    for (int iteration = 0; iteration < numIterations; ++iteration)
    {
        // Update parameters using gradient descent
        vector<P> gradient(numFeatures + 1, 0.0);

        for (int i = 0; i < m; ++i)
        {
            P error = hypothesis(X[i]) - y[i];
             

            // Update bias term (theta[0])
            gradient[0] += error;

            // Update other theta values
            for (int j = 0; j < numFeatures; ++j)
            {
                gradient[j+1] += error * X[i][j];
                //cout << i << ": "<< gradient[j] << " ";
            }
            //cout << endl;
        }

        // Update parameters using the gradient
        for (int j = 0; j <= numFeatures; ++j)
        {
            //cout << "gradiant at j = " << j << " " << gradient[j] << endl;
            //cout << "theta at j = " << j << " " << theta[j] << endl;
            theta[j] = theta[j] - (learningRate / m) * gradient[j];
            //cout << theta[j] << endl;
        }
        //cout << endl;

        // Print the cost for monitoring
        //P cost = Cost(X, y);
        //cout << "Iteration " << iteration << ", Cost: " << cost << endl;

    }
}

// Make predictions using the trained model
template <typename P>
P MultipleLinearRegression<P>::predict(const vector<P> &features) const
{
    return hypothesis(features);
}

// Get the learned parameters (theta values)
template <typename P>
vector<P> MultipleLinearRegression<P>::getTheta() const
{
    return theta;
}

// Compute the cost function
template <typename P>
P MultipleLinearRegression<P>::Cost(const vector<vector<P>> &X, const vector<P> &y) const
{
    int m = X.size(); 
    P totalError = 0;

    for (int i = 0; i < m; ++i)
    {
        P error = hypothesis(X[i]) - y[i];
        totalError += error * error;
    }

    return totalError / (2 * m);
}

/**
int main() {
    // Example usage
    // Assuming a dataset with two features and a target variable
    vector<vector<posit32_2>> X = {{1.0, 2.0}, {2.0, 3.0}, {3.0, 4.0}};
    vector<posit32_2> y = {5.0, 8.0, 11.0};

    int numFeatures = X[0].size();  // Number of features
    posit32_2 learningRate = 0.01;
    int numIterations = 1000;

    MultipleLinearRegression<posit32_2> model(numFeatures, learningRate);
    model.train(X, y, numIterations);

    // Make predictions
    vector<posit32_2> testFeatures = {4.0, 5.0};
    posit32_2 prediction = model.predict(testFeatures);

    // Display the learned parameters
    vector<posit32_2> learnedTheta = model.getTheta();
    cout << "Learned Parameters (Theta): ";
    for (posit32_2 thetaValue : learnedTheta) {
        cout << thetaValue << " ";
    }
    cout << endl;

    // Display the prediction
    cout << "Prediction: " << prediction << endl;


    return 0;
}
**/