#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <stdexcept>
#include <universal/number/posit/posit.hpp>
#include "class.h"
#include "MultLinearReg.cc"
#include "Logistic.cc"

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

class Record
{
public:
    Record() {} // Default constructor

    // Add a function to add fields dynamically as doubles/sinlges
    void addField(const string &field)
    {
        try
        {
            double value = stod(field); // Convert the string to double
            doubleFields.push_back(value);

            float f_value = stof(field); // Convert the string to single
            singleFields.push_back(f_value);
        }
        catch (const invalid_argument &e)
        {
            cerr << "Invalid field value: " << field << endl;
        }
    }

    // Accessor function to get the number of fields
    size_t getFieldCount() const
    {
        return doubleFields.size();
    }

    // Accessor function to get a specific field
    template <typename T>
    T getField(size_t index) const;

private:
    vector<double> doubleFields;
    vector<float> singleFields;
};

// Template getField so it adapts both single and double precision
template <>
double Record::getField<double>(size_t index) const
{
    if (index < doubleFields.size())
    {
        return doubleFields[index];
    }
    else
    {
        throw out_of_range("Index out of range");
    }
}

template <>
float Record::getField<float>(size_t index) const
{
    if (index < singleFields.size())
    {
        return singleFields[index];
    }
    else
    {
        throw out_of_range("Index out of range");
    }
}

// Check if the input data is numerical
bool isNumeric(const string &s)
{
    if (s.empty())
    {
        return false;
    }

    size_t i = 0;
    if (s[0] == '-')
    {
        if (s.length() == 1)
        {
            return false; // Just a hyphen is not a valid number
        }
        i = 1; // Skip the negative sign
    }

    for (; i < s.length(); i++)
    {
        if (!isdigit(s[i]) && s[i] != '.')
        {
            return false;
        }
    }

    return true;
}

// Read CSV file and convert it as a record object
vector<Record> parseCSV(const string &filename)
{
    vector<Record> records;
    ifstream file(filename);
    // int n = numCol_Ignore;
    if (!file.is_open())
    {
        cerr << "Error opening file: " << filename << endl;
        return records;
    }

    // Skip the first row (header)
    string header;
    getline(file, header);

    string line;
    while (getline(file, line))
    {
        stringstream ss(line);
        string field;
        Record record;

        /**
        // Skip the first field
        while (n != 0)
        {
            getline(ss, field, ',');
            n--;
        }
        n = numCol_Ignore;**/

        // Parse and add the remaining numerical fields as doubles
        while (getline(ss, field, ','))
        {
            if (isNumeric(field))
            {
                record.addField(field);
            }
        }

        records.push_back(record);
    }

    file.close();
    return records;
}

// Convert record object into a Posit 2d vector
template <typename P>
vector<vector<P>> convertPosit(vector<Record> records)
{
    vector<vector<P>> records_posit{};

    // Process and work with the data as double values
    for (const Record &record : records)
    {
        vector<P> temp{};
        for (size_t i = 0; i < record.getFieldCount(); ++i)
        {
            temp.push_back(record.getField<double>(i));
        }
        records_posit.push_back(temp);
    }

    return records_posit;
}

// Print Posit properties
template <size_t nbits, size_t es>
string properties()
{
    using Scalar = sw::universal::posit<nbits, es>;
    // Taking min and max number
    Scalar minpos(sw::universal::SpecificValue::minpos), maxpos(sw::universal::SpecificValue::maxpos);
    // Machine eps
    Scalar eps = numeric_limits<Scalar>::epsilon();

    cout << "nbits \tes \tminpos \t\tmaxpos \t\tepsilon \n";
    stringstream ostr;
    ostr << nbits
         << '\t'
         << es
         << '\t'
         << minpos
         << '\t'
         << maxpos
         << '\t'
         << eps
         << '\t'
         << '\n'
         << '\n';
    return ostr.str();
}

// Mean absolute error between type P and type T
template <typename P, typename T>
posit128_4 absError(vector<vector<P>> record_posit, vector<Record> records)
{
    int row = record_posit.size();
    int col = record_posit[0].size();

    posit128_4 avgError = 0;
    posit128_4 sum = 0;

    int n = 0;
    int zeros = 0;
    for (const Record &record : records)
    {
        for (size_t i = 0; i < record.getFieldCount(); ++i)
        {
            // Avoid division by 0

            posit128_4 error = abs(record.getField<T>(i) - double(record_posit[n][i]));
            sum += error;

            if (record.getField<T>(i) == 0)
            {
                zeros++;
            }
        }
        n++;
    }
    avgError = sum / (row * col - zeros);

    return avgError;
}

// Mean relative error between type P and type T
template <typename P, typename T>
posit128_4 relError(vector<vector<P>> record_posit, vector<Record> records)
{
    int row = record_posit.size();
    int col = record_posit[0].size();

    posit128_4 avgError = 0;
    posit128_4 sum = 0;

    int n = 0;
    int zeros = 0;
    for (const Record &record : records)
    {
        for (size_t i = 0; i < record.getFieldCount(); ++i)
        {
            // Avoid division by 0
            if (record.getField<T>(i) != 0)
            {
                posit128_4 error = abs((record.getField<T>(i) - double(record_posit[n][i])) / record.getField<T>(i));
                sum += error;
            }
            else if (record.getField<T>(i) == 0)
            {
                zeros++;
            }
        }
        n++;
    }
    avgError = sum / (row * col - zeros);

    return avgError;
}

// Threshold test between type P and type T
template <typename P, typename T>
double tol_test(vector<vector<P>> record_posit, vector<Record> records)
{
    // factor that changes the threshold
    double factor = 1e8;

    int row = record_posit.size();
    int col = record_posit[0].size();

    double eps_d = numeric_limits<double>::epsilon();

    int n = 0;
    int count = 0;

    double percentage = 0;

    for (const Record &record : records)
    {

        for (size_t i = 0; i < record.getFieldCount(); i++)
        {
            posit128_4 diff = abs(record.getField<T>(i) - (double)(record_posit[n][i]));
            // cout << record_posit[n][i] << endl;
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
template <typename T>
void splitVector(const vector<T> &original,
                 vector<T> &trainingSet,
                 vector<T> &testingSet,
                 double trainingRatio)
{
    // Calculate the number of elements for the training set
    size_t trainingSize = static_cast<size_t>(original.size() * trainingRatio);

    // Reserve space for the training and testing sets
    trainingSet.reserve(trainingSize);
    testingSet.reserve(original.size() - trainingSize);

    // Copy elements to the training set
    for (size_t i = 0; i < trainingSize; ++i)
    {
        trainingSet.push_back(original[i]);
    }

    // Copy remaining elements to the testing set
    for (size_t i = trainingSize; i < original.size(); ++i)
    {
        testingSet.push_back(original[i]);
    }
}

// Overload for 2D vectors
template <typename T>
void splitVector(const vector<vector<T>> &original,
                 vector<vector<T>> &trainingSet,
                 vector<vector<T>> &testingSet,
                 double trainingRatio)
{
    // Calculate the number of rows for the training set
    size_t trainingRows = static_cast<size_t>(original.size() * trainingRatio);

    // Reserve space for the training and testing sets
    trainingSet.reserve(trainingRows);
    testingSet.reserve(original.size() - trainingRows);

    // Copy rows to the training set
    for (size_t i = 0; i < trainingRows; ++i)
    {
        trainingSet.push_back(original[i]);
    }

    // Copy remaining rows to the testing set
    for (size_t i = trainingRows; i < original.size(); ++i)
    {
        testingSet.push_back(original[i]);
    }
}

// Simple linear regression, solve system with only one independent variable without iteration
template <typename P>
void linearRegression(const vector<vector<P>> &x, const vector<P> &y, P &slope, P &intercept)
{
    // Check if the input vectors have the same size
    if (x.size() != y.size())
    {
        cerr << "Error: Input vectors must have the same size." << endl;
        return;
    }

    int n = x.size();

    // Summation of x,y,x^2,y^2,xy
    posit128_4 Sx = 0.0, Sy = 0.0, Sxx = 0.0, Syy = 0.0, Sxy = 0.0;

    for (size_t i = 0; i < x.size(); ++i)
    {
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
    if (denominator != 0)
    {
        slope = (P)numerator / (P)denominator;
        intercept = (P)Sy / n - (P)slope * ((P)Sx / n);
    }
    else
    {
        cerr << "Error: Denominator is zero, linear regression is undefined." << endl;
    }
}

// Function to calculate accuracy for binary classfication
template <typename T>
double accuracy(const vector<T> &predicted, const vector<T> &actual)
{
    // Check if the sizes of predicted and actual vectors are the same
    if (predicted.size() != actual.size())
    {
        cerr << "Error: The sizes of predicted and actual values must be the same.\n";
        return 0.0;
    }

    size_t correctCount = 0;

    // Count the number of correct predictions
    for (size_t i = 0; i < predicted.size(); ++i)
    {
        if (predicted[i] == actual[i])
        {
            correctCount++;
            
        }
    }

    double result = (double)(correctCount) / predicted.size() * 100;
    // Calculate and return the accuracy
    return result;
}

// Calculate R-squared using real value y
// theta[0] is the intercept, theta[i] where i != 0 is the slope i
template <typename T>
double RSquared(const vector<vector<T>> &x, const vector<T> &y, const vector<T> theta)
{
    // Number of predictor variables
    size_t num_independent = x[0].size();

    // Calculate mean of y
    double mean_y = 0.0;
    for (size_t i = 0; i < y.size(); ++i)
    {
        mean_y += (double)y[i];
    }
    mean_y /= y.size();

    // Calculate Total Sum of Squares (SST)
    double SST = 0.0;
    for (size_t i = 0; i < y.size(); ++i)
    {
        SST += ((double)y[i] - mean_y) * ((double)y[i] - mean_y);
    }

    double SSR = 0.0;
    for (size_t i = 0; i < y.size(); ++i)
    {
        double predicted_y = (double)theta[0]; // Intercept term

        for (size_t j = 0; j < num_independent; ++j)
        {
            predicted_y += (double)theta[j + 1] * (double)(x[i][j]);
        }

        SSR += ((double)y[i] - predicted_y) * ((double)y[i] - predicted_y);
    }

    // Calculate R-squared
    double rSquared = 1.0 - (SSR / SST);
    return rSquared;
}

template <typename P>
void exportToCSV(const vector<vector<P>> &data, const string &filename)
{
    // Open the file for writing
    ofstream outputFile(filename);

    // Check if the file is open
    if (!outputFile.is_open())
    {
        cerr << "Error opening file for exporting: " << filename << endl;
        return;
    }

    outputFile << setprecision(15);

    // Iterate through the rows of the 2D vector and write to the file
    for (const auto &row : data)
    {
        for (size_t i = 0; i < row.size(); ++i)
        {
            outputFile << row[i];
            if (i < row.size() - 1)
            {
                outputFile << ","; // Add a comma between values
            }
        }
        outputFile << endl; // Move to the next line after each row
    }

    // Close the file
    outputFile.close();

    cout << "Posit format data has been successfully exported to " << filename << endl;
}

// Print the errors corresponding to a precision
template <typename P, typename T>
void printResultPosit(vector<Record> records, string filename, int size, int es, int models, double learningRate, int numIterations)
{

    vector<vector<P>> records_posit = convertPosit<P>(records);

    cout << setprecision(15);
    posit128_4 absErr = absError<P, T>(records_posit, records);
    cout << "Posit vs Double absolute error: " << absErr << endl;

    posit128_4 relErr = relError<P, T>(records_posit, records);
    cout << "Posit vs Double relative error: " << relErr << endl;

    double tol_test_pass = tol_test<P, T>(records_posit, records);
    cout << "Posit vs Double tolerance test pass rate: " << setprecision(4) << tol_test_pass << "%" << endl;

    // Export posit file
    string fname = filename.erase(filename.length() - 4);

    // Remove address in name
    if (fname.length() >= 13 && fname.substr(0, 13) == "TestingFiles/")
    {
        fname.erase(0, 13);
    }

    stringstream name;
    name << "Generated_files/Generated_" << fname << '_' << to_string(size) << '_' << to_string(es) << ".csv";

    exportToCSV(records_posit, name.str());

    cout << endl;

    /*************/
    vector<vector<P>> x(records_posit.size(), vector<P>(records_posit[0].size() - 1));
    vector<P> y(records_posit.size());

    if (models == 1)
    {
        // Simple Linear regression
        if (records_posit[0].size() == 2)
        {

            for (size_t i = 0; i < records_posit.size(); ++i)
            {

                y[i] = records_posit[i][0];
                for (size_t j = 0; j < records_posit[0].size() - 1; ++j)
                {
                    x[i][j] = records_posit[i][1];
                }
            }

            cout << "Using simple linear regression: " << endl;
            P intercept;
            P slope;
            // Perform linear regression
            linearRegression<P>(x, y, slope, intercept);
            cout << setprecision(10) << "slope: " << slope << ", "
                 << "intercept: " << intercept << endl;

            vector<P> theta = {intercept, slope};

            double rSquared = RSquared(x, y, theta);
            cout << "R-squared: " << rSquared << endl
                 << endl;
        }

        // Multiple linear regression
        cout << "Using multiple linear regression: " << endl;
        for (size_t i = 0; i < records_posit.size(); ++i)
        {
            // y is at the first column
            y[i] = records_posit[i][0];
            for (size_t j = 0; j < records_posit[0].size() - 1; ++j)
            {
                x[i][j] = records_posit[i][j + 1];
            }
        }

        int numFeatures = x[0].size();
        //double learningRate = 0.00002;
        //int numIterations = 1000;
        MultipleLinearRegression<P> model(numFeatures, learningRate);
        model.train(x, y, numIterations);

        cout << setprecision(10);
        vector<P> learnedTheta = model.getTheta();
        cout << "Learned Parameters (Theta): ";
        for (P thetaValue : learnedTheta)
        {
            cout << thetaValue << " ";
        }
        cout << endl;

        double rSquared = RSquared(x, y, learnedTheta);

        cout << "R-squared: " << rSquared << endl;
        cout << endl;
    }

    // Binary Classification
    else if (models == 2)
    {
        vector<int> labels(records_posit.size());
        // Logistic regression
        cout << "Using logistic regression: " << endl;
        for (size_t i = 0; i < records_posit.size(); ++i)
        {
            // labels is at the first column
            labels[i] = (int)records_posit[i][0];
            for (size_t j = 0; j < records_posit[0].size() - 1; ++j)
            {
                x[i][j] = records_posit[i][j + 1];
            }
        }

        vector<int> trainingLabel, testingLabel;
        vector<vector<P>> trainingSet, testingSet;
        double trainingRatio = 0.7;

        // Create training set and testing set
        splitVector(labels, trainingLabel, testingLabel, trainingRatio);
        splitVector(x, trainingSet, testingSet, trainingRatio);

        //double learningRate = 0.000002;
        //int numIterations = 100;
        LogisticRegression<P> logistic_model(trainingSet, trainingLabel, learningRate);

        // Train the Logistic Regression
        logistic_model.train(numIterations);

        vector<int> result;
        for (const auto &row : testingSet)
        {
            P prediction = logistic_model.predict(row);
            int resultLabels = logistic_model.roundProbability(prediction,0.3);
            result.push_back(resultLabels);  
        }
        cout << "Use " << trainingRatio << " total data for training the model." << endl;
        cout << "Accuracy is " << accuracy(result, testingLabel) << "%" << endl;
    }
}

template <typename P, typename T>
void printResultIEEE(vector<Record> records, int models, double learningRate, int numIterations)
{

    vector<vector<P>> records_IEEE = convertPosit<P>(records);

    cout << setprecision(15);
    posit128_4 absErr = absError<P, T>(records_IEEE, records);
    cout << "Absolute error: " << absErr << endl;

    posit128_4 relErr = relError<P, T>(records_IEEE, records);
    cout << "Relative error: " << relErr << endl;

    double tol_test_pass = tol_test<P, T>(records_IEEE, records);
    cout << "Tolerance test pass rate: " << setprecision(4) << tol_test_pass << "%" << endl;

    cout << endl;
    /*************/
    vector<vector<P>> x(records_IEEE.size(), vector<P>(records_IEEE[0].size() - 1));
    vector<P> y(records_IEEE.size());

    if (models == 1)
    {
        // Simple Linear regression
        if (records_IEEE[0].size() == 2)
        {

            for (size_t i = 0; i < records_IEEE.size(); ++i)
            {
                // y is at the first column
                y[i] = records_IEEE[i][0];
                for (size_t j = 0; j < records_IEEE[0].size() - 1; ++j)
                {
                    x[i][j] = records_IEEE[i][1];
                }
            }

            cout << "Linear regression: " << endl;
            P intercept;
            P slope;
            // Perform linear regression
            linearRegression<P>(x, y, slope, intercept);
            cout << setprecision(10) << "slope: " << slope << ", "
                 << "intercept: " << intercept << endl;

            vector<P> theta = {intercept, slope};

            double rSquared = RSquared(x, y, theta);
            cout << "R-squared: " << rSquared << endl
                 << endl;
        }

        // Multiple linear regression
        cout << "Using multiple linear regression: " << endl;
        for (size_t i = 0; i < records_IEEE.size(); ++i)
        {
            // y is at the first column
            y[i] = records_IEEE[i][0];
            for (size_t j = 0; j < records_IEEE[0].size() - 1; ++j)
            {
                x[i][j] = records_IEEE[i][j + 1];
            }
        }

        int numFeatures = x[0].size();
        //double learningRate = 0.00002;
        //int numIterations = 1000;
        MultipleLinearRegression<P> model(numFeatures, learningRate);
        model.train(x, y, numIterations);

        cout << setprecision(10);
        vector<P> learnedTheta = model.getTheta();
        cout << "Learned Parameters (Theta): ";
        for (P thetaValue : learnedTheta)
        {
            cout << thetaValue << " ";
        }
        cout << endl;

        double rSquared = RSquared(x, y, learnedTheta);
        cout << "R-squared: " << rSquared << endl;
    }

    // Binary Classification
    else if (models == 2)
    {
        vector<int> labels(records_IEEE.size());
        // Logistic regression
        cout << "Using logistic regression: " << endl;
        for (size_t i = 0; i < records_IEEE.size(); ++i)
        {
            // labels is at the first column
            labels[i] = records_IEEE[i][0];
            for (size_t j = 0; j < records_IEEE[0].size() - 1; ++j)
            {
                x[i][j] = records_IEEE[i][j + 1];
            }
        }

        vector<int> trainingLabel, testingLabel;
        vector<vector<P>> trainingSet, testingSet;
        double trainingRatio = 0.7;

        // Create training set and testing set
        splitVector(labels, trainingLabel, testingLabel, trainingRatio);
        splitVector(x, trainingSet, testingSet, trainingRatio);

        //double learningRate = 0.000002;
        //int numIterations = 100;
        LogisticRegression<P> logistic_model(trainingSet, trainingLabel, learningRate);

        // Train the Logistic Regression
        logistic_model.train(numIterations);

        vector<int> result;
        for (const auto &row : testingSet)
        {
            P prediction = logistic_model.predict(row);
            int resultLabels = logistic_model.roundProbability(prediction,0.3);
            result.push_back(resultLabels);  
        }
        cout << "Use " << trainingRatio << " total data for training the model." << endl;
        cout << "Accuracy is " << accuracy(result, testingLabel) << "%" << endl;
    }
}

int main(int argc, char **argv)
{
    if (argc < 7)
    {
        cout << "usage: " << argv[0] << " <filename> <Posit size> <Exponent bit size> <Regression/Classification> <number of iteration> <learning rate>\n";
        cout << "<filename> are usually in the format of \"TestingFiles/filename.csv\"\n";
        cout << "<Posit size> can be selected between 16 or 32 \n";
        cout << "<Exponent bit size> can be selected between 0,1,2 3 or 4 \n";
        cout << "<Regression/Classification> can be decided by using 0: DON'T DO any model, or 1: do regression, or 2: do binary classfication \n\n";
        cout << "Example: .\\main TestingFiles/Marketing_new.csv 32 2 1 100 0.002\n";
        return 1;
    }

    string filename = argv[1];
    // int numCol_Ignore = stoi(argv[2]);
    vector<Record> records = parseCSV(filename);
    cout << "Parse completed: " << filename << endl;

    size_t totalData = records[0].getFieldCount() * records.size();
    cout << "The total number of data is: " << totalData << endl;

    cout << endl;

    unsigned int p_size = stoi(argv[2]);
    unsigned int p_es = stoi(argv[3]);
    int models = stoi(argv[4]);

    
    int numIterations = stof(argv[5]);
    double learningRate = stof(argv[6]);

    posit128_4 absErr;
    posit128_4 relErr;

    if (p_size == 16)
    {
        if (p_es == 0)
        {
            cout << properties<16, 0>();
            printResultPosit<posit16_0, double>(records, filename, p_size, p_es, models,learningRate,numIterations);
        }

        else if (p_es == 1)
        {
            cout << properties<16, 1>();
            printResultPosit<posit16_1, double>(records, filename, p_size, p_es, models,learningRate,numIterations);
        }
        else if (p_es == 2)
        {
            cout << properties<16, 2>();
            printResultPosit<posit16_2, double>(records, filename, p_size, p_es, models,learningRate,numIterations);
        }
        else if (p_es == 3)
        {
            cout << properties<16, 3>();
            printResultPosit<posit16_3, double>(records, filename, p_size, p_es, models,learningRate,numIterations);
        }

        else if (p_es == 4)
        {
            cout << properties<16, 4>();
            printResultPosit<posit16_4, double>(records, filename, p_size, p_es, models,learningRate,numIterations);
        }
        else
        {
            cerr << "Unsupported p_es" << endl;
            return 1;
        }
    }
    else if (p_size == 32)
    {
        if (p_es == 0)
        {
            cout << properties<32, 0>();
            printResultPosit<posit32_0, double>(records, filename, p_size, p_es, models,learningRate,numIterations);
        }
        else if (p_es == 1)
        {
            cout << properties<32, 1>();
            printResultPosit<posit32_1, double>(records, filename, p_size, p_es, models,learningRate,numIterations);
        }
        else if (p_es == 2)
        {
            cout << properties<32, 2>();
            printResultPosit<posit32_2, double>(records, filename, p_size, p_es, models,learningRate,numIterations);
        }
        else if (p_es == 3)
        {
            cout << properties<32, 3>();
            printResultPosit<posit32_3, double>(records, filename, p_size, p_es, models,learningRate,numIterations);
        }
        else if (p_es == 4)
        {
            cout << properties<32, 4>();
            printResultPosit<posit32_4, double>(records, filename, p_size, p_es, models,learningRate,numIterations);
        }
        else
        {
            cerr << "Unsupported p_es" << endl;
            return 1;
        }
    }
    else
    {
        cerr << "Unsupported p_size" << endl;
        return 1;
    }

    cout << "====================================" << endl;
    cout << "Single vs Double: " << endl;
    printResultIEEE<float, double>(records, models,learningRate,numIterations);

    cout << "====================================" << endl;
    cout << "Double vs Double: " << endl;
    printResultIEEE<double, double>(records, models,learningRate,numIterations);

    // vector<vector<float>> records_float = convertPosit<float>(records);

    // posit128_4 absErr_float;
    // posit128_4 relErr_float;

    /**
    Errors<float, double>(records_float, records,absErr_float,relErr_float);
    cout << "Single precision: Absolute error: " << absErr_float << endl;

    cout << "Single precision: Relative error: " << relErr_float << endl;

    posit128_4 absErr_diff = absErr_float / absErr;
    cout << "Precision difference based on Absolute error: " << absErr_diff << endl;

    posit128_4 relErr_diff = relErr_float / relErr;
    cout << "recision difference based on Relative error: " << relErr_diff << endl;
    **/

    return 0;
}
