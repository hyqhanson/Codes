#include <iostream>
#include <vector>

#include "class.h"

using namespace std;

inline double
nearestPowerOfTwo(double num)
{
  double lowerPowerOfTwo = pow(2, floor(log(num) / log(2)));
  double upperPowerOfTwo = pow(2, ceil(log(num) / log(2)));
  // Determine which power of two is closer to the original number
  if (num - lowerPowerOfTwo < upperPowerOfTwo - num) {
    return lowerPowerOfTwo;
  } else {
    return upperPowerOfTwo;
  }
}

// template <typename P>
// void findMaxMin(vector<vector<P>> record_posit, P &Max, P &Min) {
//   // Initialize max and min to the first element of the matrix
//   Max = record_posit[0][0];
//   Min = record_posit[0][0];

//   // Iterate over each vector in the matrix
//   for (const auto &row : record_posit) {
//     // Find the maximum and minimum elements in the current vector
//     P maxInRow = *std::max_element(row.begin(), row.end());
//     P minInRow = *std::min_element(row.begin(), row.end());

//     // Update global maximum and minimum if needed
//     Max = std::max(Max, maxInRow);
//     Min = std::min(Min, minInRow);
//   }
// }

// template <typename P>
// void minMaxScaling(vector<vector<P>> record_posit, P Max, P Min,
//                    vector<vector<P>> &new_record) {

//   // A loop to copy elements of old vector into new vector by Iterative
//   method for (size_t i = 0; i < record_posit.size(); ++i) {
//     for (size_t j = 0; j < record_posit[0].size(); ++j) {
//       new_record[i][j] = record_posit[i][j];
//       new_record[i][j] = (record_posit[i][j] - Min) / (Max - Min);
//     }
//   }
// }

// Function to perform min-max scaling for a single column
template<typename P>
void
minMaxScaleColumn(vector<P>& column)
{
  // Find the minimum and maximum values in the column
  auto min_max = minmax_element(column.begin(), column.end());
  P min_val = *min_max.first;
  P max_val = *min_max.second;

  P div = (max_val - min_val);

  // Perform min-max scaling for each value in the column
  for (P& value : column) {
    value = (value) / div;
  }

  cout << "The min is " << min_val << ", the max is " << max_val << endl;
}

// Function to perform min-max scaling for a 2D vector column by column
template<typename P>
void
minMaxScale2D(vector<vector<P>>& data)
{
  // Iterate over each column of the 2D vector
  for (size_t col = 0; col < data[0].size(); ++col) {
    // Extract the column
    vector<P> column;
    for (size_t row = 0; row < data.size(); ++row) {
      column.push_back(data[row][col]);
    }
    // Scale the column
    minMaxScaleColumn(column);
    // Update the original data with the scaled column
    for (size_t row = 0; row < data.size(); ++row) {
      data[row][col] = column[row];
    }
  }
}

/************************/

// Function to calculate the mean of a vector
template<typename P>
P
calculateMean(const vector<P>& data)
{
  P sum = 0.0;
  for (const auto& value : data) {
    sum += value;
  }
  return sum / data.size();
}

// Function to calculate the standard deviation of a vector
template<typename P>
P
calculateStdDev(const vector<P>& data, P mean)
{
  P sumSquaredDiff = 0.0;
  for (const auto& value : data) {
    sumSquaredDiff += pow(value - mean, 2);
  }
  return sqrt(sumSquaredDiff / data.size());
}

// Function to standardize a vector (z-score normalization)
template<typename P>
void
standardize(vector<P>& data)
{
  P mean = calculateMean(data);
  P stdDev = calculateStdDev(data, mean);
  cout << mean << " " << stdDev << endl;
  vector<P> standardizedData;
  for (auto& value : data) {
    value = (value - mean) / stdDev;
  }
 
}

// Function to perform min-max scaling for a 2D vector column by column
template<typename P>
void
standard2D(vector<vector<P>>& data)
{
  // Iterate over each column of the 2D vector
  for (size_t col = 0; col < data[0].size(); ++col) {
    // Extract the column
    vector<P> column;
    for (size_t row = 0; row < data.size(); ++row) {
      column.push_back(data[row][col]);
    }
    // Scale the column
    standardize(column);
    // Update the original data with the scaled column
    for (size_t row = 0; row < data.size(); ++row) {
      data[row][col] = column[row];
    }
  }
}