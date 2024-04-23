#include <iostream>
#include <vector>

#include "class.h"

using namespace std;

inline double nearestPowerOfTwo(double num)
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

template <typename P>
void findMaxMin(vector<vector<P>> record_posit, P &Max, P &Min)
{
    // Initialize max and min to the first element of the matrix
    Max = record_posit[0][0];
    Min = record_posit[0][0];

    // Iterate over each vector in the matrix
    for (const auto &row : record_posit)
    {
        // Find the maximum and minimum elements in the current vector
        P maxInRow = *std::max_element(row.begin(), row.end());
        P minInRow = *std::min_element(row.begin(), row.end());

        // Update global maximum and minimum if needed
        Max = std::max(Max, maxInRow);
        Min = std::min(Min, minInRow);
    }
}

template <typename P>
void minMaxScaling(vector<vector<P>> record_posit, P Max, P Min, vector<vector<P>> &new_record)
{

    // A loop to copy elements of old vector into new vector by Iterative method
    for (size_t i = 0; i < record_posit.size(); ++i)
    {
        for (size_t j = 0; j < record_posit[0].size(); ++j)
        {
            new_record[i][j] = record_posit[i][j];
            new_record[i][j] = (record_posit[i][j] - Min) / (Max - Min);
        }
    }
}