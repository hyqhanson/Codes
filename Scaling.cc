#include <iostream>
#include <vector>
#include <cmath>
#include "class.h"

using namespace std;

inline double
nearestPowerOfTwo(double num)
{
  double lowerPowerOfTwo = pow(2, floor(log(num) / log(2)));
  double upperPowerOfTwo = pow(2, ceil(log(num) / log(2)));
  // Determine which power of two is closer to the original number
  if (num - lowerPowerOfTwo < upperPowerOfTwo - num)
  {
    return lowerPowerOfTwo;
  }
  else
  {
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
template <typename P>
void minMaxScaleColumn(vector<P> &column, vector<vector<P>> &scalingFactor)
{
  // Find the minimum and maximum values in the column
  auto min_max = minmax_element(column.begin(), column.end());
  P min_val = *min_max.first;
  P max_val = *min_max.second;

  P div = (max_val - min_val);

  // Perform min-max scaling for each value in the column
  if (div != 0)
  {
    for (P &value : column)
    {
      value = (value - min_val) / div;
    }
  }

  vector<P> Factor = {min_val, max_val};

  scalingFactor.push_back(Factor);

  cout << "The min is " << min_val << ", the max is " << max_val << endl;
}

// Function to perform min-max scaling for a 2D vector column by column
template <typename P>
void minMaxScale2D(vector<vector<P>> &data, vector<vector<P>> &scalingFactor)
{
  // Iterate over each column of the 2D vector
  for (size_t col = 0; col < data[0].size(); ++col)
  {
    // Extract the column
    vector<P> column;
    for (size_t row = 0; row < data.size(); ++row)
    {
      column.push_back(data[row][col]);
    }
    // Scale the column
    minMaxScaleColumn(column, scalingFactor);
    // Update the original data with the scaled column
    for (size_t row = 0; row < data.size(); ++row)
    {
      data[row][col] = column[row];
    }
  }
}

/************************/

// Function to calculate the mean of a vector
template <typename P>
P calculateMean(const vector<P> &data)
{
  P sum = (P)0.0;
  int count = 0;
  for (const auto &value : data)
  {
    P next_sum = sum + value;

    // Avoid overflow
    if (isinf(next_sum))
    {
      break;
    }
    // else if (!isinf(next_sum) && next_sum != sum)
    // {
    //   sum = next_sum;
    //   count++;
    // }
    else if (!isinf(next_sum))
    {
      sum = next_sum;
      count++;
    }
  }
  if (count == 0)
  {
    count = 1;
  }

  return (P)(sum / count);
}

// Function to calculate the standard deviation of a vector
template <typename P>
P calculateStdDev(const vector<P> &data, P mean)
{
  P sumSquaredDiff = P(0.0);
  int count = 0;
  for (const auto &value : data)
  {
    P next_sumSquaredDiff = sumSquaredDiff + (P)pow(value - mean, 2);

    // Avoid overflow
    if (isinf(next_sumSquaredDiff))
    {
      break;
    }
    // else if (!isinf(next_sumSquaredDiff) && next_sumSquaredDiff != sumSquaredDiff)
    // {
    //   sumSquaredDiff = next_sumSquaredDiff;
    //   count++;
    // }
    else if (!isinf(next_sumSquaredDiff))
    {
      sumSquaredDiff = next_sumSquaredDiff;
      count++;
    }
  }

  if (count == 0)
  {
    count = 1;
  }
  return (P)(sqrt(sumSquaredDiff / count));
}

// Function to standardize a vector (z-score normalization)
template <typename P>
void standardize(vector<P> &data, vector<vector<P>> &scalingFactor)
{
  P mean = calculateMean(data);
  P stdDev = calculateStdDev(data, mean);
  cout << "Mean: " << mean << ",stdDev: " << stdDev << endl;
  vector<P> standardizedData;
  if (stdDev != 0)
  {
    for (auto &value : data)
    {
      value = (value - mean) / (stdDev);
    }
  }

  vector<P> Factor = {mean, stdDev};
  scalingFactor.push_back(Factor);
}

// Function to perform min-max scaling for a 2D vector column by column
template <typename P>
void standard2D(vector<vector<P>> &data,
                vector<vector<P>> &scalingFactor)
{
  // Iterate over each column of the 2D vector
  for (size_t col = 0; col < data[0].size(); ++col)
  {
    // Extract the column
    vector<P> column;
    for (size_t row = 0; row < data.size(); ++row)
    {
      column.push_back(data[row][col]);
    }
    // Scale the column
    standardize(column, scalingFactor);
    // Update the original data with the scaled column
    for (size_t row = 0; row < data.size(); ++row)
    {
      data[row][col] = column[row];
    }
  }
}

// // Function to standardize a vector (z-score normalization)
// template <typename P>
// void standardize(vector<P> &data, const P &mean, const P &stdDev)
// {
//   cout << "Mean: " << mean << ", stdDev: " << stdDev << endl;
//   if (stdDev != 0)
//   {
//     for (auto &value : data)
//     {
//       value = (value - mean) / stdDev;
//     }
//   }
// }

// // Function to perform z-score normalization for a 2D vector column by column
// template <typename P>
// void standard2D(vector<vector<P>> &data, const vector<vector<P>> &scalingFactor)
// {
//   // Check if scalingFactor size matches the number of columns in data
//   if (scalingFactor.size() != data[0].size())
//   {
//     cerr << "Error: scalingFactor size does not match the number of columns in data." << endl;
//     return;
//   }

//   // Iterate over each column of the 2D vector
//   for (size_t col = 0; col < data[0].size(); ++col)
//   {
//     // Extract the mean and stdDev for the current column from scalingFactor
//     P mean = scalingFactor[col][0];
//     P stdDev = scalingFactor[col][1];

//     // Extract the column
//     vector<P> column;
//     for (size_t row = 0; row < data.size(); ++row)
//     {
//       column.push_back(data[row][col]);
//     }

//     // Scale the column using the provided mean and stdDev
//     standardize(column, mean, stdDev);

//     // Update the original data with the scaled column
//     for (size_t row = 0; row < data.size(); ++row)
//     {
//       data[row][col] = column[row];
//     }
//   }
// }
