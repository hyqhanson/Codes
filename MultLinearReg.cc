#include "class.h"
#include <iostream>
#include <random>
#include <vector>

using namespace std;

template <typename P>
MultipleLinearRegression<P>::MultipleLinearRegression(int numFeatures,
                                                      double learningRate,
                                                      unsigned seed)
    : numFeatures(numFeatures), learningRate(learningRate), seed(seed)
{

  // Initialize theta with random number in [-1,1]
  theta.resize(numFeatures + 1, (P)0.0);

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
P MultipleLinearRegression<P>::hypothesis(const vector<P> &features) const
{
  P result = theta[0]; // Initialize with the bias term
  // cout << "result adding theta0 " << result << endl;
  // Multiply each feature by its corresponding theta and sum them up
  for (int i = 0; i < numFeatures; ++i)
  {
    result += theta[i + 1] * features[i];
    // cout << theta[i + 1] << endl;
    // cout << "x" << i << " " << features[i] << endl;
    // cout << "result complete" << result << endl;
  }
  // cout << endl;

  return result;
}

// Train the model using gradient descent
template <typename P>
void MultipleLinearRegression<P>::train(const vector<vector<P>> &x,
                                        const vector<P> &y,
                                        int numIterations)
{
  int m = x.size(); // Number of training examples

  double max_cost = numeric_limits<double>::max();

  vector<double> cost_list;
  for (int iteration = 0; iteration < numIterations; ++iteration)
  {
    // Update parameters using gradient descent
    vector<P> gradient(numFeatures + 1, (P)0.0);

    for (int i = 0; i < m; ++i)
    {
      P error = hypothesis(x[i]) - y[i];

      // if (i == 0)
      // {
      //   cout << x[0][0] << " " << x[0][1] << endl;
      //   cout << "Theta " << theta[0] << " " << theta[1] << " " << theta[2] << endl;
      //   cout << "y_0 = " << (double)hypothesis(x[i]) << endl;
      //   cout << "y = " << y[i] << endl;
      //   for (int j = 0; j <= numFeatures; ++j)
      //   {
      //     cout << "theta = " << theta[j] << " ";
      //   }
      //   cout << endl;
      //   cout << "error: " << error << endl;
      // }

      // Update bias term (theta[0])
      P next_grad0 = gradient[0] + error;
      // Avoid overflow
      if (isinf(next_grad0))
      {
        break;
      }
      else
      {
        gradient[0] = next_grad0;
      }

      // Update other theta values
      for (int j = 0; j < numFeatures; ++j)
      {
        P next_grad = gradient[j + 1] + error * x[i][j];

        // Avoid overflow
        if (isinf(next_grad))
        {
          break;
        }
        else
        {
          gradient[j + 1] = next_grad;
        }

        // cout << j + 1 << ": " << gradient[j + 1] << endl;
      }
    }

    // Update parameters using the gradient
    for (int j = 0; j <= numFeatures; ++j)
    {
      // cout << "gradiant at j = " << j << " " << gradient[j] << endl;

      theta[j] = theta[j] - (learningRate / m) * gradient[j];
      // cout << "theta at " << j << " = " << theta[j] << endl;
      // cout << "gradiant: " << gradient[j] << ", result: " << (learningRate / m) * (double)gradient[j] << endl;
      // cout << theta[j] << endl;
    }
    // cout << endl;

    // Print the cost for monitoring
    double cost = Cost(x, y);

    cost_list.push_back(cost);

    if (cost >= max_cost || max_cost - cost < 1e-3)
    {
      cout << "Convergence stops at iteration " << iteration << endl;
      break;
    }

    max_cost = cost;
  }

  // plot(cost_list);
}

// Train the model using stochastic gradient descent
template <typename P>
void MultipleLinearRegression<P>::train_stoch(const vector<vector<P>> &x,
                                              const vector<P> &y,
                                              int numIterations)
{
  int m = x.size(); // Number of training examples

  // Use a random number generator for shuffling the data
  // random_device rd;
  // mt19937 gen(rd());
  mt19937 gen(seed);

  double max_cost = numeric_limits<double>::max();
  vector<double> cost_list;
  for (int iteration = 0; iteration < numIterations; ++iteration)
  {
    // Shuffle the dataset
    vector<size_t> indices(m);
    iota(indices.begin(), indices.end(), 0);
    shuffle(indices.begin(), indices.end(), gen);

    // Update parameters using stochastic gradient descent
    for (int i = 0; i < m; ++i)
    {
      size_t index = indices[i];

      P error = hypothesis(x[index]) - y[index];

      // Update bias term (theta[0])
      theta[0] -= learningRate * error;

      // Update other theta values
      for (int j = 0; j < numFeatures; ++j)
      {
        theta[j + 1] -= learningRate * error * x[index][j];
      }
    }

    // Print the cost for monitoring
    double cost = Cost(x, y);

    cost_list.push_back(cost);

    if (cost >= max_cost || max_cost - cost < 1e-6)
    {
      cout << "Convergence stops at iteration " << iteration << endl;
      break;
    }

    max_cost = cost;
  }

  // plot(cost_list);
}

// Make predictions using the trained model
template <typename P>
P MultipleLinearRegression<P>::predict(const vector<P> &features) const
{
  return hypothesis(features);
}

// Get the learned parameters (theta values)
template <typename P>
vector<P>
MultipleLinearRegression<P>::getTheta() const
{
  return theta;
}

// Compute the cost function
template <typename P>
double
MultipleLinearRegression<P>::Cost(const vector<vector<P>> &x,
                                  const vector<P> &y) const
{
  int m = x.size();
  double totalError = 0;

  for (int i = 0; i < m; ++i)
  {
    double error = (double)hypothesis(x[i]) - (double)y[i];

    totalError += error * error;
  }

  return totalError / (2 * m);
}

// Compute MSE
template <typename P>
double
MultipleLinearRegression<P>::MSE(const vector<vector<P>> &x,
                                 const vector<double> &y) const
{
  double mse = 0.0;
  for (size_t i = 0; i < x.size(); ++i)
  {
    double error = (double)hypothesis(x[i]) - (double)y[i];
    mse += error * error;
  }

  mse /= x.size(); // Calculate mean

  return mse;
}
