#include <iostream>
#include <random>
#include <vector>

#include "class.h"

using namespace std;

template <typename P>
MultipleLinearRegression<P>::MultipleLinearRegression(int numFeatures,
                                                      double learningRate,
                                                      unsigned seed)
    : numFeatures(numFeatures), learningRate(learningRate) {

  // Initialize theta with random number in [-1,1]
  theta.resize(numFeatures + 1, 0.0);

  // mt19937 gen(seed);
  // uniform_real_distribution<double> dist(0.0, 1.0);
  // for (int i = 0; i < numFeatures + 1; ++i) {
  //   theta[i] = dist(gen);
  //   cout << theta[i] << " ";
  // }
  // cout << endl;
}

// Hypothesis function for multiple linear regression
template <typename P>
P MultipleLinearRegression<P>::hypothesis(const vector<P> &features) const {
  P result = theta[0]; // Initialize with the bias term

  // Multiply each feature by its corresponding theta and sum them up
  for (int i = 0; i < numFeatures; ++i) {
    result += theta[i + 1] * features[i];
  }

  return result;
}

// Train the model using gradient descent
template <typename P>
void MultipleLinearRegression<P>::train(const vector<vector<P>> &x,
                                        const vector<P> &y, int numIterations) {
  int m = x.size(); // Number of training examples

  P max_cost = numeric_limits<double>::max();

  for (int iteration = 0; iteration < numIterations; ++iteration) {
    // Update parameters using gradient descent
    vector<P> gradient(numFeatures + 1, 0.0);

    for (int i = 0; i < m; ++i) {
      P error = hypothesis(x[i]) - y[i];

      // Update bias term (theta[0])
      gradient[0] += error;

      // Update other theta values
      for (int j = 0; j < numFeatures; ++j) {
        gradient[j + 1] += error * x[i][j];
        // cout << i << ": "<< gradient[j] << " ";
      }
      // cout << endl;
    }

    // Update parameters using the gradient
    for (int j = 0; j <= numFeatures; ++j) {
      // cout << "gradiant at j = " << j << " " << gradient[j] << endl;
      // cout << "theta at j = " << j << " " << theta[j] << endl;
      theta[j] = theta[j] - (learningRate / m) * gradient[j];
      // cout << theta[j] << endl;
    }
    // cout << endl;

    // Print the cost for monitoring
    P cost = Cost(x, y);
    // cout << "Cost: " << cost << endl;
    if (cost >= max_cost) {
      cout << "Failed to converge at iteration " << iteration << endl;
      break;
    }

    max_cost = cost;
  }
}

// Make predictions using the trained model
template <typename P>
P MultipleLinearRegression<P>::predict(const vector<P> &features) const {
  return hypothesis(features);
}

// Get the learned parameters (theta values)
template <typename P> vector<P> MultipleLinearRegression<P>::getTheta() const {
  return theta;
}

// Compute the cost function
template <typename P>
double MultipleLinearRegression<P>::Cost(const vector<vector<P>> &x,
                                    const vector<P> &y) const {
  int m = x.size();
  double totalError = 0;

  for (int i = 0; i < m; ++i) {
    double error = (double)hypothesis(x[i]) - (double)y[i];
    totalError += error * error;
  }

  return totalError / (2 * m);
}

// Compute MSE
template <typename P>
double MultipleLinearRegression<P>::MSE(const vector<vector<P>> &x,
                                        const vector<double> &y) const {
  double mse = 0.0;
  for (size_t i = 0; i < x.size(); ++i) {
    double error = (double)hypothesis(x[i]) - (double)y[i];
    mse += error * error;
  }

  mse /= x.size(); // Calculate mean

  return mse;
}
