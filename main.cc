#include "Logistic.cc"
#include "MultLinearReg.cc"
#include "MultLinearReg_CG.cc"
#include "Scaling.cc"
#include "class.h"
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <universal/number/posit/posit.hpp>
#include <vector>

using namespace sw::universal;
using namespace std;

typedef posit<16, 0> posit16_0;
typedef posit<16, 1> posit16_1;
typedef posit<16, 2> posit16_2;
typedef posit<16, 3> posit16_3;
typedef posit<16, 4> posit16_4;

typedef posit<32, 0> posit32_0;
typedef posit<32, 1> posit32_1;
typedef posit<32, 2> posit32_2;
typedef posit<32, 3> posit32_3;
typedef posit<32, 4> posit32_4;

typedef posit<64, 0> posit64_0;
typedef posit<64, 1> posit64_1;
typedef posit<64, 2> posit64_2;
typedef posit<64, 3> posit64_3;
typedef posit<64, 4> posit64_4;

typedef posit<128, 4> posit128_4;

// Check if the input data is numerical
bool
isNumeric(const string& s)
{
  if (s.empty()) {
    return false;
  }

  size_t i = 0;
  if (s[0] == '-') {
    if (s.length() == 1) {
      return false; // Just a hyphen is not a valid number
    }
    i = 1; // Skip the negative sign
  }

  for (; i < s.length(); i++) {
    if (!isdigit(s[i]) && s[i] != '.') {
      return false;
    }
  }

  return true;
}

// Read CSV file and convert it as a record object
vector<vector<double>>
parseCSV(const string& filename)
{
  vector<vector<double>> Records;
  ifstream file(filename);
  // int n = numCol_Ignore;
  if (!file.is_open()) {
    cerr << "Error opening file: " << filename << endl;
    return Records;
  }

  // Skip the first row (header)
  string header;
  getline(file, header);

  string line;
  while (getline(file, line)) {
    stringstream ss(line);
    string field;
    vector<double> record;

    /**
    // Skip the first field
    while (n != 0)
    {
        getline(ss, field, ',');
        n--;
    }
    n = numCol_Ignore;**/

    // Parse and add the remaining numerical fields as doubles
    while (getline(ss, field, ',')) {
      if (isNumeric(field)) {
        double value = stod(field); // Convert the string to double
        record.push_back(value);
      }
    }

    Records.push_back(record);
  }

  file.close();
  return Records;
}

// Convert record object into a Posit 2d vector
template<typename P>
vector<vector<P>>
convertPosit(vector<vector<double>> Records)
{
  vector<vector<P>> Records_posit{};

  // Process and work with the data as double values
  for (const auto& record : Records) {
    vector<P> temp{};
    for (size_t i = 0; i < record.size(); ++i) {
      temp.push_back(record[i]);
    }
    Records_posit.push_back(temp);
  }

  return Records_posit;
}

// Print Posit properties
template<size_t nbits, size_t es>
string
properties()
{
  using Scalar = sw::universal::posit<nbits, es>;
  // Taking min and max number
  Scalar minpos(sw::universal::SpecificValue::minpos),
    maxpos(sw::universal::SpecificValue::maxpos);
  // Machine eps
  Scalar eps = numeric_limits<Scalar>::epsilon();

  cout << "nbits \tes \tminpos \t\tmaxpos \t\tepsilon \n";
  stringstream ostr;
  ostr << nbits << '\t' << es << '\t' << minpos << '\t' << maxpos << '\t' << eps
       << '\t' << '\n'
       << '\n';
  return ostr.str();
}

// Mean absolute error between type P and type T
template<typename P, typename T>
posit128_4
absError(vector<vector<P>> Record_posit, vector<vector<double>> Records)
{
  int row = Record_posit.size();
  int col = Record_posit[0].size();

  posit128_4 avgError = 0;
  posit128_4 sum = 0;

  int n = 0;
  int zeros = 0;
  for (const auto& record : Records) {
    for (size_t i = 0; i < record.size(); ++i) {
      posit128_4 error = abs(record[i] - double(Record_posit[n][i]));
      // cout << record[i] << " " << Record_posit[n][i]<<endl;
      sum += error;

      if (record[i] == 0) {
        zeros++;
      }
    }
    n++;
  }
  avgError = sum / (row * col - zeros);

  return avgError;
}

// Overwrite mean absolute error just for the result
template<typename P, typename T>
posit128_4
absError(vector<P> Record_posit, vector<vector<double>> Records)
{
  posit128_4 avgError = 0;
  posit128_4 sum = 0;

  int zeros = 0;
  for (size_t i = 0; i < Records.size(); ++i) {
    posit128_4 error = abs(Records[i][0] - double(Record_posit[i]));
    sum += error;
    if (Records[i][0] == 0) {
      zeros++;
    }
  }
  avgError = sum / Records.size();

  return avgError;
}

// Mean relative error between type P and type T
template<typename P, typename T>
posit128_4
relError(vector<vector<P>> Record_posit, vector<vector<double>> Records)
{
  int row = Record_posit.size();
  int col = Record_posit[0].size();

  posit128_4 avgError = 0;
  posit128_4 sum = 0;

  int n = 0;
  int zeros = 0;
  for (const auto& record : Records) {
    for (size_t i = 0; i < record.size(); ++i) {
      // Avoid division by 0
      if (record[i] != 0) {
        posit128_4 error =
          abs((record[i] - double(Record_posit[n][i])) / record[i]);
        sum += error;
      } else if (record[i] == 0) {
        zeros++;
      }
    }
    n++;
  }
  avgError = sum / (row * col - zeros);

  return avgError;
}

// Overwrite mean relative error just for the result
template<typename P, typename T>
posit128_4
relError(vector<P> Record_posit, vector<vector<double>> Records)
{

  posit128_4 avgError = 0;
  posit128_4 sum = 0;

  int zeros = 0;
  for (size_t i = 0; i < Records.size(); ++i) {
    // Avoid division by 0
    if (Records[i][0] != 0) {
      posit128_4 error =
        abs((Records[i][0] - double(Record_posit[i])) / Records[i][0]);
      sum += error;
    } else if (Records[i][0] == 0) {
      zeros++;
    }
  }
  avgError = sum / Records.size();

  return avgError;
}

// Threshold test between type P and type T
template<typename P, typename T>
double
tol_test(vector<vector<P>> Record_posit, vector<vector<double>> Records)
{
  // factor that changes the threshold
  double factor = 1e8;

  int row = Record_posit.size();
  int col = Record_posit[0].size();

  double eps_d = numeric_limits<double>::epsilon();

  int n = 0;
  int count = 0;

  double percentage = 0;

  for (const auto& record : Records) {

    for (size_t i = 0; i < record.size(); i++) {
      posit128_4 diff = abs(record[i] - (double)(Record_posit[n][i]));
      // cout << Record_posit[n][i] << endl;
      // cout << diff << "    " << eps_d * factor << endl;
      if (diff <= eps_d * factor)
        count++;
    }
    n++;
  }

  if (count != 0)
    percentage = ((double)count / (double)(row * col)) * 100;

  return percentage;
}

// Generic function to split a vector into training and testing sets
template<typename T>
void
splitVector(const vector<T>& original,
            vector<T>& trainingSet,
            vector<T>& testingSet,
            double trainingRatio)
{
  // Calculate the number of elements for the training set
  size_t trainingSize = static_cast<size_t>(original.size() * trainingRatio);

  // Reserve space for the training and testing sets
  trainingSet.reserve(trainingSize);
  testingSet.reserve(original.size() - trainingSize);

  // Copy elements to the training set
  for (size_t i = 0; i < trainingSize; ++i) {
    trainingSet.push_back(original[i]);
  }

  // Copy remaining elements to the testing set
  for (size_t i = trainingSize; i < original.size(); ++i) {
    testingSet.push_back(original[i]);
  }
}

// Overload for 2D vectors
template<typename T>
void
splitVector(const vector<vector<T>>& original,
            vector<vector<T>>& trainingSet,
            vector<vector<T>>& testingSet,
            double trainingRatio)
{
  // Calculate the number of rows for the training set
  size_t trainingRows = static_cast<size_t>(original.size() * trainingRatio);

  // Reserve space for the training and testing sets
  trainingSet.reserve(trainingRows);
  testingSet.reserve(original.size() - trainingRows);

  // Copy rows to the training set
  for (size_t i = 0; i < trainingRows; ++i) {
    trainingSet.push_back(original[i]);
  }

  // Copy remaining rows to the testing set
  for (size_t i = trainingRows; i < original.size(); ++i) {
    testingSet.push_back(original[i]);
  }
}

// Simple linear regression, solve system with only one independent variable
// without iteration
template<typename P>
void
linearRegression(const vector<vector<P>>& x,
                 const vector<P>& y,
                 P& slope,
                 P& intercept)
{
  // Check if the input vectors have the same size
  if (x.size() != y.size()) {
    cerr << "Error: Input vectors must have the same size." << endl;
    return;
  }

  int n = x.size();

  // Summation of x,y,x^2,y^2,xy
  posit128_4 Sx = 0.0, Sy = 0.0, Sxx = 0.0, Syy = 0.0, Sxy = 0.0;

  for (size_t i = 0; i < x.size(); ++i) {
    Sx += x[i][0];
    Sy += y[i];
    Sxx += x[i][0] * x[i][0];
    Syy += y[i] * y[i];
    Sxy += x[i][0] * y[i];
  }

  // Slope = (n*Sxy-Sx*Sy)/(n*Sxx-Sx*Sx)
  // Intercept = Sy/n - slope * (Sx/n)
  posit128_4 numerator = n * Sxy - Sx * Sy;
  posit128_4 denominator = n * Sxx - Sx * Sx;
  if (denominator != 0) {
    slope = (P)numerator / (P)denominator;
    intercept = (P)Sy / n - (P)slope * ((P)Sx / n);
  } else {
    cerr << "Error: Denominator is zero, linear regression is undefined."
         << endl;
  }
}

// Function to calculate accuracy for binary classfication
template<typename T>
double
accuracy(const vector<T>& predicted, const vector<T>& actual)
{
  // Check if the sizes of predicted and actual vectors are the same
  if (predicted.size() != actual.size()) {
    cerr << "Error: The sizes of predicted and actual values must be the "
            "same.\n";
    return 0.0;
  }

  size_t correctCount = 0;

  // Count the number of correct predictions
  for (size_t i = 0; i < predicted.size(); ++i) {
    if (predicted[i] == actual[i]) {
      correctCount++;
    }
  }

  double result = (double)(correctCount) / predicted.size() * 100;
  // Calculate and return the accuracy
  return result;
}

// Calculate R-squared using real value y
// theta[0] is the intercept, theta[i] where i != 0 is the slope i
template<typename T>
double
RSquared(const vector<vector<T>>& x, const vector<T>& y, const vector<T> theta)
{
  // Number of predictor variables
  size_t num_independent = x[0].size();

  // Calculate mean of y
  double mean_y = 0.0;
  for (size_t i = 0; i < y.size(); ++i) {
    mean_y += (double)y[i];
    // cout << (double)y[i] << " ";
  }
  mean_y /= y.size();

  // Calculate Total Sum of Squares (SST)
  double SST = 0.0;
  for (size_t i = 0; i < y.size(); ++i) {
    SST += ((double)y[i] - mean_y) * ((double)y[i] - mean_y);
  }

  // Calculate Sum of Squares of Residuals (SSR)
  double SSR = 0.0;
  for (size_t i = 0; i < y.size(); ++i) {
    double predicted_y = (double)theta[0]; // Intercept term

    for (size_t j = 0; j < num_independent; ++j) {
      predicted_y += (double)theta[j + 1] * (double)(x[i][j]);
    }

    SSR += ((double)y[i] - predicted_y) * ((double)y[i] - predicted_y);
  }

  // Calculate R-squared
  double rSquared = 1.0 - (SSR / SST);
  return rSquared;
}

template<typename P>
void
exportToCSV(const vector<vector<P>>& data, const string& filename)
{
  // Open the file for writing
  ofstream outputFile(filename);

  // Check if the file is open
  if (!outputFile.is_open()) {
    cerr << "Error opening file for exporting: " << filename << endl;
    return;
  }

  outputFile << setprecision(15);

  // Iterate through the rows of the 2D vector and write to the file
  for (const auto& row : data) {
    for (size_t i = 0; i < row.size(); ++i) {
      outputFile << row[i];
      if (i < row.size() - 1) {
        outputFile << ","; // Add a comma between values
      }
    }
    outputFile << endl; // Move to the next line after each row
  }

  // Close the file
  outputFile.close();

  cout << "Posit format data has been successfully exported to " << filename
       << endl;
}

template<typename P>
void
printRecord(const vector<vector<P>>& data)
{
  for (size_t i = 0; i < data.size(); i++) {
    for (size_t j = 0; j < data[0].size(); j++) {
      cout << data[i][j] << " ";
    }
    cout << endl;
  }
}

// Print the errors corresponding to a precision
template<typename P, typename T>
void
printResultPosit(vector<vector<double>> Records,
                 vector<vector<double>> scaled_Records,
                 string filename,
                 int size,
                 int es,
                 int models,
                 double learningRate,
                 int numIterations,
                 unsigned seed,
                 int scaled,
                 vector<vector<double>> scalingFactor)
{
  vector<vector<P>> Records_posit;
  vector<double> yd(Records.size());
  if (scaled == 0) {
    Records_posit = convertPosit<P>(Records);
    posit128_4 absErr = absError<P, T>(Records_posit, Records);
    cout << "Posit vs Double absolute error: " << absErr << endl;

    posit128_4 relErr = relError<P, T>(Records_posit, Records);
    cout << "Posit vs Double relative error: " << relErr << endl;

    double tol_test_pass = tol_test<P, T>(Records_posit, Records);
    cout << "Posit vs Double tolerance test pass rate: " << setprecision(4)
         << tol_test_pass << "%" << endl;

    for (size_t i = 0; i < Records.size(); ++i) {
      yd[i] = Records[i][0];
    }
  } else {
    Records_posit = convertPosit<P>(scaled_Records);
    posit128_4 absErr = absError<P, T>(Records_posit, scaled_Records);
    cout << "Posit vs Double absolute error: " << absErr << endl;

    posit128_4 relErr = relError<P, T>(Records_posit, scaled_Records);
    cout << "Posit vs Double relative error: " << relErr << endl;

    double tol_test_pass = tol_test<P, T>(Records_posit, scaled_Records);
    cout << "Posit vs Double tolerance test pass rate: " << setprecision(4)
         << tol_test_pass << "%" << endl;

    for (size_t i = 0; i < Records.size(); ++i) {
      yd[i] = scaled_Records[i][0];
    }
  }

  cout << setprecision(15);
  // P Max = 0;
  // P Min = 0;
  // findMaxMin<P>(Records_posit, Max, Min);
  // cout << "Max value: " << Max << ", "
  //      << "Min value:" << Min << endl;

  // Create a scaling record
  // vector<vector<P>>
  // scaled_Record(Records_posit.size(),vector<P>(Records_posit[0].size()));
  // minMaxScaling(Records_posit, Max, Min, scaled_Record);

  // Export posit file
  string fname = filename.erase(filename.length() - 4);

  // Remove address in name
  if (fname.length() >= 13 && fname.substr(0, 13) == "TestingFiles/") {
    fname.erase(0, 13);
  }

  stringstream name;
  name << "Generated_files/Generated_" << fname << '_' << to_string(size) << '_'
       << to_string(es) << ".csv";

  exportToCSV(Records_posit, name.str());

  cout << endl;

  /*************/
  vector<vector<P>> x(Records_posit.size(),
                      vector<P>(Records_posit[0].size() - 1));
  vector<P> y(Records_posit.size());

  if (models == 1) {
    // Simple Linear regression
    if (Records_posit[0].size() == 2) {

      for (size_t i = 0; i < Records_posit.size(); ++i) {

        y[i] = Records_posit[i][0];
        for (size_t j = 0; j < Records_posit[0].size() - 1; ++j) {
          x[i][j] = Records_posit[i][1];
        }
      }

      cout << "Using simple linear regression: " << endl;
      P intercept;
      P slope;
      // Perform linear regression
      linearRegression<P>(x, y, slope, intercept);
      cout << setprecision(10) << "slope: " << slope << ", "
           << "intercept: " << intercept << endl;

      vector<P> theta = { intercept, slope };

      double rSquared = RSquared(x, y, theta);
      cout << "R-squared: " << rSquared << endl;

      double mse = 0.0;
      for (size_t i = 0; i < x.size(); ++i) {
        double error = (double)(slope * x[i][0] + intercept) - (double)y[i];
        mse += error * error;
      }

      mse /= x.size();

      cout << "MSE: " << mse << endl << endl;
    }

    // Multiple linear regression
    cout << "Using multiple linear regression: " << endl;
    for (size_t i = 0; i < Records_posit.size(); ++i) {
      // y is at the first column
      y[i] = Records_posit[i][0];
      for (size_t j = 0; j < Records_posit[0].size() - 1; ++j) {
        x[i][j] = Records_posit[i][j + 1];
      }
    }

    int numFeatures = x[0].size();
    // double learningRate = 0.00002;
    // int numIterations = 1000;
    MultipleLinearRegression<P> model(numFeatures, learningRate, seed);
    // model.train(x, y, numIterations);
    model.train_stoch(x, y, numIterations);

    cout << setprecision(15);
    vector<P> learnedTheta = model.getTheta();
    cout << "Learned Parameters (Theta): ";
    for (P thetaValue : learnedTheta) {
      cout << thetaValue << " ";
    }
    cout << endl;

    double rSquared = RSquared(x, y, learnedTheta);
    double mse = model.MSE(x, yd);

    if (scaled != 0) {
      vector<double> result;
      // scaling factor 1 = min or mean
      double factor1 = scalingFactor[0][0];
      // scaling factor 2 = max or StdDev
      double factor2 = (scalingFactor[0][1]);

      double mse_restore = 0.0;

      // min-max
      if (scaled == 1) {
        for (size_t i = 0; i < x.size(); ++i) {
          result.push_back((double)model.hypothesis(x[i]));
          result[i] = result[i] * (factor2 - factor1) + factor1;
          double error = result[i] - Records[i][0];
          mse_restore += error * error;
        }
      }
      // standard
      else if (scaled == 2) {
        for (size_t i = 0; i < x.size(); ++i) {
          result.push_back((double)model.hypothesis(x[i]));
          result[i] = result[i] * factor2 + factor1;
          double error = result[i] - Records[i][0];

          mse_restore += error * error;
        }
      }

      mse_restore /= x.size();
      cout << "MSE restored: " << mse_restore << endl;
      posit128_4 absolute_error = absError<T, T>(result, Records);
      cout << "Absolute error: " << absolute_error << endl;
      posit128_4 relative_error = relError<T, T>(result, Records);
      cout << "Relative error: " << relative_error << endl;
    }
    cout << "R-squared: " << rSquared << endl;
    cout << "MSE: " << mse << endl;

    cout << endl;
  }

  // Binary Classification
  else if (models == 2) {
    vector<int> labels(Records_posit.size());
    // Logistic regression
    cout << "Using logistic regression: " << endl;
    for (size_t i = 0; i < Records_posit.size(); ++i) {
      // labels is at the first column
      labels[i] = (int)Records_posit[i][0];
      for (size_t j = 0; j < Records_posit[0].size() - 1; ++j) {
        x[i][j] = Records_posit[i][j + 1];
      }
    }

    vector<int> trainingLabel, testingLabel;
    vector<vector<P>> trainingSet, testingSet;
    double trainingRatio = 0.7;

    // Create training set and testing set
    splitVector(labels, trainingLabel, testingLabel, trainingRatio);
    splitVector(x, trainingSet, testingSet, trainingRatio);

    // double learningRate = 0.000002;
    // int numIterations = 100;
    LogisticRegression<P> logistic_model(
      trainingSet, trainingLabel, learningRate);

    // Train the Logistic Regression
    logistic_model.train(numIterations);

    vector<int> result;
    for (const auto& row : testingSet) {
      P prediction = logistic_model.predict(row);
      int resultLabels = logistic_model.roundProbability(prediction, 0.3);
      result.push_back(resultLabels);
    }
    cout << "Use " << trainingRatio << " total data for training the model."
         << endl;
    cout << "Accuracy is " << accuracy(result, testingLabel) << "%" << endl;
  }

  // CG
  else if (models == 3) {
    cout << "Using multiple linear regression with CG: " << endl;
    for (size_t i = 0; i < Records_posit.size(); ++i) {
      // y is at the first column
      y[i] = Records_posit[i][0];
      for (size_t j = 0; j < Records_posit[0].size() - 1; ++j) {
        x[i][j] = Records_posit[i][j + 1];
      }
    }

    // Parameters for CG
    double tolerance = 1e-6;

    // Solve multiple linear regression using ,Conjugate Gradient
    vector<P> coefficients =
      conjugateGradient<P>(x, y, numIterations, tolerance);

    // Print coefficients
    for (size_t i = 0; i < Records[0].size(); ++i) {
      cout << "Coefficient " << i << ": " << coefficients[i] << endl;
    }
    cout << endl;
  }
}

template<typename P, typename T>
void
printResultIEEE(vector<vector<double>> Records,
                vector<vector<double>> scaled_Records,
                int models,
                double learningRate,
                int numIterations,
                unsigned seed,
                int scaled,
                vector<vector<double>> scalingFactor)
{

  vector<vector<P>> Records_IEEE;
  vector<double> yd(Records.size());
  if (scaled == 0) {
    Records_IEEE = convertPosit<P>(Records);
    posit128_4 absErr = absError<P, T>(Records_IEEE, Records);
    cout << "Posit vs Double absolute error: " << absErr << endl;

    posit128_4 relErr = relError<P, T>(Records_IEEE, Records);
    cout << "Posit vs Double relative error: " << relErr << endl;

    double tol_test_pass = tol_test<P, T>(Records_IEEE, Records);
    cout << "Posit vs Double tolerance test pass rate: " << setprecision(4)
         << tol_test_pass << "%" << endl;

    for (size_t i = 0; i < Records.size(); ++i) {
      yd[i] = Records[i][0];
    }
  } else {
    Records_IEEE = convertPosit<P>(scaled_Records);
    posit128_4 absErr = absError<P, T>(Records_IEEE, scaled_Records);
    cout << "Posit vs Double absolute error: " << absErr << endl;

    posit128_4 relErr = relError<P, T>(Records_IEEE, scaled_Records);
    cout << "Posit vs Double relative error: " << relErr << endl;

    double tol_test_pass = tol_test<P, T>(Records_IEEE, scaled_Records);
    cout << "Posit vs Double tolerance test pass rate: " << setprecision(4)
         << tol_test_pass << "%" << endl;

    for (size_t i = 0; i < Records.size(); ++i) {
      yd[i] = scaled_Records[i][0];
    }
  }

  cout << setprecision(15);

  cout << endl;
  /*************/
  vector<vector<P>> x(Records_IEEE.size(),
                      vector<P>(Records_IEEE[0].size() - 1));
  vector<P> y(Records_IEEE.size());

  if (models == 1) {
    // Simple Linear regression
    if (Records_IEEE[0].size() == 2) {

      for (size_t i = 0; i < Records_IEEE.size(); ++i) {
        // y is at the first column
        y[i] = Records_IEEE[i][0];
        for (size_t j = 0; j < Records_IEEE[0].size() - 1; ++j) {
          x[i][j] = Records_IEEE[i][1];
        }
      }

      cout << "Linear regression: " << endl;
      P intercept;
      P slope;
      // Perform linear regression
      linearRegression<P>(x, y, slope, intercept);
      cout << setprecision(10) << "slope: " << slope << ", "
           << "intercept: " << intercept << endl;

      vector<P> theta = { intercept, slope };

      double rSquared = RSquared(x, y, theta);
      cout << "R-squared: " << rSquared << endl << endl;
    }

    // Multiple linear regression
    cout << "Using multiple linear regression: " << endl;
    for (size_t i = 0; i < Records_IEEE.size(); ++i) {
      // y is at the first column
      y[i] = Records_IEEE[i][0];
      for (size_t j = 0; j < Records_IEEE[0].size() - 1; ++j) {
        x[i][j] = Records_IEEE[i][j + 1];
      }
    }

    int numFeatures = x[0].size();
    // double learningRate = 0.00004;
    // int numIterations = 1000;
    MultipleLinearRegression<P> model(numFeatures, learningRate, seed);
    // model.train(x, y, numIterations);
    model.train_stoch(x, y, numIterations);

    cout << setprecision(15);
    vector<P> learnedTheta = model.getTheta();
    cout << "Learned Parameters (Theta): ";

    for (P thetaValue : learnedTheta) {
      cout << thetaValue << " ";
    }
    cout << endl;

    double rSquared = RSquared(x, y, learnedTheta);
    double mse = model.MSE(x, yd);

    if (scaled != 0) {
      vector<P> result;
      // scaling factor 1 = min or mean
      P factor1 = scalingFactor[0][0];
      // scaling factor 2 = max or StdDev
      P factor2 = (scalingFactor[0][1]);

      double mse_restore = 0.0;
      // min-max
      if (scaled == 1) {
        for (size_t i = 0; i < x.size(); ++i) {
          result.push_back(model.hypothesis(x[i]));
          result[i] = result[i] * (factor2 - factor1) + factor1;
          double error = (double)result[i] - Records[i][0];
          mse_restore += error * error;
        }
      }
      // standard
      else if (scaled == 2) {
        for (size_t i = 0; i < x.size(); ++i) {
          result.push_back((double)model.hypothesis(x[i]));
          result[i] = result[i] * factor2 + factor1;

          double error = result[i] - Records[i][0];

          mse_restore += error * error;
        }
      }
      mse_restore /= x.size();
      cout << "MSE restored: " << mse_restore << endl;
      posit128_4 absolute_error = absError<P, T>(result, Records);
      cout << "Absolute error: " << absolute_error << endl;
      posit128_4 relative_error = relError<P, T>(result, Records);
      cout << "Relative error: " << relative_error << endl;
    }
    cout << "R-squared: " << rSquared << endl;
    cout << "MSE: " << mse << endl;

    cout << endl;
  }

  // Binary Classification
  else if (models == 2) {
    vector<int> labels(Records_IEEE.size());
    // Logistic regression
    cout << "Using logistic regression: " << endl;
    for (size_t i = 0; i < Records_IEEE.size(); ++i) {
      // labels is at the first column
      labels[i] = Records_IEEE[i][0];
      for (size_t j = 0; j < Records_IEEE[0].size() - 1; ++j) {
        x[i][j] = Records_IEEE[i][j + 1];
      }
    }

    vector<int> trainingLabel, testingLabel;
    vector<vector<P>> trainingSet, testingSet;
    double trainingRatio = 0.7;

    // Create training set and testing set
    splitVector(labels, trainingLabel, testingLabel, trainingRatio);
    splitVector(x, trainingSet, testingSet, trainingRatio);

    // double learningRate = 0.000002;
    // int numIterations = 100;
    LogisticRegression<P> logistic_model(
      trainingSet, trainingLabel, learningRate);

    // Train the Logistic Regression
    logistic_model.train(numIterations);

    vector<int> result;
    for (const auto& row : testingSet) {
      P prediction = logistic_model.predict(row);
      int resultLabels = logistic_model.roundProbability(prediction, 0.3);
      result.push_back(resultLabels);
    }
    cout << "Use " << trainingRatio << " total data for training the model."
         << endl;
    cout << "Accuracy is " << accuracy(result, testingLabel) << "%" << endl;
  }

  // CG
  else if (models == 3) {
    cout << "Using multiple linear regression with CG: " << endl;
    for (size_t i = 0; i < Records_IEEE.size(); ++i) {
      // y is at the first column
      y[i] = Records_IEEE[i][0];
      for (size_t j = 0; j < Records_IEEE[0].size() - 1; ++j) {
        x[i][j] = Records_IEEE[i][j + 1];
      }
    }

    // Parameters for CG
    double tolerance = 1e-6;

    // Solve multiple linear regression using Conjugate Gradient
    vector<P> coefficients = conjugateGradient(x, y, numIterations, tolerance);

    // Print coefficients
    for (size_t i = 0; i < Records[0].size(); ++i) {
      cout << "Coefficient " << i << ": " << coefficients[i] << endl;
    }
  }
}

int
main(int argc, char** argv)
{
  unsigned seed = chrono::system_clock::now().time_since_epoch().count();

  if (argc < 7) {
    cout << "usage: " << argv[0]
         << " <filename> <Posit size> <Scaling> "
            "<Regression/Classification> <number of iteration> <learning "
            "rate>\n";
    cout << "<filename> are usually in the format of "
            "\"TestingFiles/filename.csv\"\n";
    cout << "<Posit size> can be selected between 16 or 32 \n";
    cout << "<Scaling> can be selected between 0 (without scale), 1 (min-max "
            "scale) \n";
    cout
      << "<Regression/Classification> can be decided by using 0: DON'T DO "
         "any model, or 1: do regression, or 2: do binary classfication \n\n";
    cout << "Example: ./main TestingFiles/Marketing.csv 32 2 1 1000 0.00002\n";
    return 1;
  }

  string filename = argv[1];
  // int numCol_Ignore = stoi(argv[2]);

  // Storing file into Records
  vector<vector<double>> Records = parseCSV(filename);

  cout << setprecision(15);
  // double Max = 0;
  // double Min = 0;
  // findMaxMin<double>(Records, Max, Min);
  // cout << "Max value: " << Max << ", "
  //      << "Min value:" << Min << endl;

  cout << "Parse completed: " << filename << endl;

  size_t totalData = Records[0].size() * Records.size();
  cout << "The total number of data is: " << totalData << endl;

  cout << endl;

  unsigned int p_size = stoi(argv[2]);
  unsigned int scale = stoi(argv[3]);
  int models = stoi(argv[4]);

  int numIterations = stof(argv[5]);
  double learningRate = stof(argv[6]);

  // Scaling
  vector<vector<double>> scaled_Records(Records.size(),
                                        vector<double>(Records[0].size(), 0.0));
  vector<vector<double>> scalingFactor;
  if (scale == 1) {
    cout << "Using min-max" << endl;
    minMaxScale2D(Records, scaled_Records, scalingFactor);
  } else if (scale == 2) {
    cout << "Using standard" << endl;
    standard2D(Records, scaled_Records, scalingFactor);
  }

  posit128_4 absErr;
  posit128_4 relErr;

#define noScale(size, r)                                                       \
  {                                                                            \
    printf("Posit<%d,%d> \n", size, r);                                        \
    printResultPosit<posit<size, r>, double>(Records,                          \
                                             scaled_Records,                   \
                                             filename,                         \
                                             p_size,                           \
                                             r,                                \
                                             models,                           \
                                             learningRate,                     \
                                             numIterations,                    \
                                             seed,                             \
                                             scale,                            \
                                             scalingFactor);                   \
  }

  if (p_size == 16) {
    noScale(16, 0);
    noScale(16, 1);
    noScale(16, 2);
    noScale(16, 3);
    noScale(16, 4);

  } else if (p_size == 32) {
    noScale(32, 0);
    noScale(32, 1);
    noScale(32, 2);
    noScale(32, 3);
    noScale(32, 4);
  } else {
    cerr << "Unsupported p_size" << endl;
    return 1;
  }

  cout << "====================================" << endl;
  cout << "Single vs Double: " << endl;
  printResultIEEE<float, double>(Records,
                                 scaled_Records,
                                 models,
                                 learningRate,
                                 numIterations,
                                 seed,
                                 scale,
                                 scalingFactor);

  cout << "====================================" << endl;
  cout << "Double vs Double: " << endl;
  printResultIEEE<double, double>(Records,
                                  scaled_Records,
                                  models,
                                  learningRate,
                                  numIterations,
                                  seed,
                                  scale,
                                  scalingFactor);

  // vector<vector<float>> Records_float = convertPosit<float>(Records);

  // posit128_4 absErr_float;
  // posit128_4 relErr_float;

  /**
  Errors<float, double>(Records_float, Records,absErr_float,relErr_float);
  cout << "Single precision: Absolute error: " << absErr_float << endl;

  cout << "Single precision: Relative error: " << relErr_float << endl;

  posit128_4 absErr_diff = absErr_float / absErr;
  cout << "Precision difference based on Absolute error: " << absErr_diff <<
  endl;

  posit128_4 relErr_diff = relErr_float / relErr;
  cout << "recision difference based on Relative error: " << relErr_diff <<
  endl;
  **/

  return 0;
}
