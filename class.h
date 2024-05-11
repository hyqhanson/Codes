#ifndef CLASS_H
#define CLASS_H

#include <universal/number/posit/posit.hpp>
#include <vector>

template<typename P>
class MultipleLinearRegression
{
private:
  int numFeatures;      // Number of features
  std::vector<P> theta; // Parameters
  double learningRate;  // Learning rate for gradient descent

public:
  MultipleLinearRegression(int numFeatures, double learningRate, unsigned seed);

  // Hypothesis function for multiple linear regression
  P hypothesis(const std::vector<P>& features) const;

  // Train the model using gradient descent
  void train(const std::vector<std::vector<P>>& x,
             const std::vector<P>& y,
             int numIterations);

  // Train the model using stochastic gradient descent
  void train_stoch(const std::vector<std::vector<P>>& x,
                   const std::vector<P>& y,
                   int numIterations);

  // Make predictions using the trained model
  P predict(const std::vector<P>& features) const;

  // Get the learned parameters (theta values)
  std::vector<P> getTheta() const;

  double Cost(const std::vector<std::vector<P>>& x,
              const std::vector<P>& y) const;

  double MSE(const std::vector<std::vector<P>>& x,
             const std::vector<double>& y) const;
};

template<typename P>
class LogisticRegression
{
private:
  std::vector<std::vector<P>> data; // Input data
  std::vector<int> labels;          // Class labels (0 or 1)
  std::vector<P> weights;           // Parameters (weights)
  P bias;                           // Bias term
  P learningRate;                   // Learning rate for optimization

public:
  LogisticRegression(const std::vector<std::vector<P>>& input_data,
                     const std::vector<int>& input_labels,
                     double learning_rate);

  // Logistic function (sigmoid)
  P sigmoid(P z) const;

  // Logistic Regression prediction function
  P predict(const std::vector<P>& input) const;

  double crossEntropyLoss() const;

  // Train the Logistic Regression using gradient descent
  void train(int numIterations);

  // Display the learned parameters (weights and bias)
  void displayParameters() const;

  int roundProbability(P probability, P threshold);
};

// min-max
inline double
nearestPowerOfTwo(double num);

template<typename P>
void
minMaxScaleColumn(std::vector<P>& column,
                  std::vector<std::vector<P>>& scalingFactor);

template<typename P>
void
minMaxScale2D(std::vector<std::vector<P>>& data,
              std::vector<std::vector<P>>& scaled_data,
              std::vector<std::vector<P>>& scalingFactor);

// Standard
template<typename P>
P
calculateMean(const std::vector<P>& data);

template<typename P>
P
calculateStdDev(const std::vector<P>& data, P mean);

template<typename P>
void
standardize(std::vector<P>& data,
            std::vector<std::vector<P>>& scalingFactor);

template<typename P>
void
standard2D(std::vector<std::vector<P>>& data,
           std::vector<std::vector<P>>& scaled_data,
           std::vector<std::vector<P>>& scalingFactor);

// CG method
template<typename T>
T
dotProduct(const std::vector<T>& a, const std::vector<T>& b);

template<typename T>
std::vector<T>
multiplyMatrixVector(const std::vector<std::vector<T>>& A,
                     const std::vector<T>& x);

template<typename T>
std::vector<T>
conjugateGradient(const std::vector<std::vector<T>>& X,
                  const std::vector<T>& y,
                  size_t maxIterations,
                  double tolerance);

// Plotting
void
plot(std::vector<double>& points);

#endif
