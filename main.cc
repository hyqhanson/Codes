#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <stdexcept>
#include <universal/number/posit/posit.hpp>

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
        while (getline(ss, field, ',') )
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

/** Calculate average absolute and relative error of record in IEEE double compare to in posits
 *
 *
 **/
/**
template <typename P, typename T>
void Errors(const vector<vector<P>> record_posit, const vector<Record> records, posit128_4& avg_absError, posit128_4& avg_relError)
{
    int row = record_posit.size();
    int col = record_posit[0].size();

    posit128_4 abs_sum = 0;
    posit128_4 rel_sum = 0;

    posit128_4 abs_error;
    posit128_4 rel_error;

    int n = 0;
    int zeros = 0;
    for (const Record &record : records)
    {
        for (size_t i = 0; i < record.getFieldCount(); ++i)
        {
            if (record.getField<T>(i) != 0)
            {
            abs_error = abs(record.getField<T>(i) - double(record_posit[n][i]));
            rel_error = abs(abs_error / record.getField<T>(i));

            abs_sum += abs_error;
            rel_sum += rel_error;
            }
            else if (record.getField<T>(i) == 0)
            {
                zeros++;
            }
        }
        n++;
    }

    avg_absError = abs_sum / (row * col - zeros);

    avg_relError = rel_sum / (row * col - zeros);


}
**/

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

/** Calculate relative error of record in IEEE double and in posits
 *
 *
 **/
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

template <typename P, typename T>
double tol_test(vector<vector<P>> record_posit, vector<Record> records, double factor)
{
    int row = record_posit.size();
    int col = record_posit[0].size();

    double eps_d = numeric_limits<double>::epsilon();

    int n = 0;
    int count = 0;

    double percentage = 0;

    for (const Record &record : records)
    {
        for (size_t i = 0; i < record.getFieldCount(); ++i)
        {
            // Avoid division by 0
            if (record.getField<T>(i) != 0)
            {
                posit128_4 diff = abs(record.getField<T>(i) - double(record_posit[n][i]));
                if (diff <= eps_d * factor)
                    count++;
            }
        }
        n++;
    }

    if (count != 0)
        percentage = ((double)count / (double)(row * col)) * 100;

    return percentage;
}

template <typename P>
void linearRegression(const vector<P> &x, const vector<P> &y, double &slope, double &intercept)
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
        Sx += x[i];
        Sy += y[i];
        Sxx += x[i] * x[i];
        Syy += y[i] * y[i];
        Sxy += x[i] * y[i];
    }

   
    // Slope = (n*Sxy-Sx*Sy)/(n*Sxx-Sx*Sx)
    // Intercept = Sy/n - slope * (Sx/n)
    posit128_4 numerator = n * Sxy - Sx * Sy;
    posit128_4 denominator = n * Sxx - Sx * Sx;
    if (denominator != 0)
    {
        slope = (double)numerator / (double)denominator;
        intercept = (double)Sy / n - slope * ((double)Sx / n);
    }
    else
    {
        cerr << "Error: Denominator is zero, linear regression is undefined." << endl;
    }
}

template <typename P>
double RSquared(const vector<P> &x, const vector<P> &y, double slope, double intercept)
{
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

    // Calculate Sum of Squared Residuals (SSR)
    double SSR = 0.0;
    for (size_t i = 0; i < x.size(); ++i)
    {
        P predicted_y = slope * x[i] + intercept;
        SSR += ((double)y[i] - (double)predicted_y) * ((double)y[i] - (double)predicted_y);
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
void printResultPosit(vector<Record> records, string filename, int size, int es)
{

    vector<vector<P>> records_posit = convertPosit<P>(records);

    // posit128_4 absErr;
    // posit128_4 relErr;
    // Errors<P, T>(records_posit, records, absErr, relErr);

    /**
   // Print dataset in Posit for testing
   for (const vector<P> p : records_posit)
   {
       for (int i = 0; i < p.size(); i++)
       {
           cout << setprecision(10) << double(p[i]) << ", ";
       }
       cout << endl;
   }
   cout << endl;

   // Print dataset in IEEE for testing
   for (const Record &record : records)
   {
       // Example: Print all fields
       for (size_t i = 0; i < record.getFieldCount(); ++i)
       {
           cout << record.getField<T>(i) << ", ";
       }
       cout << endl;
   }**/
    cout << setprecision(15);
    posit128_4 absErr = absError<P, T>(records_posit, records);
    cout << "Posit vs Double absolute error: " << absErr << endl;

    posit128_4 relErr = relError<P, T>(records_posit, records);
    cout << "Posit vs Double relative error: " << relErr << endl;

    double tol_test_pass = tol_test<P, T>(records_posit, records, 1e6);
    cout << "Posit vs Double tolerance test pass rate: " << setprecision(3) << tol_test_pass << "%" << endl;

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
    // Linear regression
    vector<P> x(records_posit.size());
    vector<P> y(records_posit.size());

    for (int i = 0; i < records_posit.size(); i++)
    {
        x[i] = records_posit[i][0];
        y[i] = records_posit[i][1];
    }

    double slope, intercept;
    // Perform linear regression
    linearRegression<P>(x, y, slope, intercept);
    cout << setprecision(10) << "slope: " << slope << ", "
         << "intercept: " << intercept << endl;

    double rSquared = RSquared(x, y, slope, intercept);
    cout << "R-squared: " << rSquared << endl;

    cout << endl;
}

template <typename P, typename T>
void printResultIEEE(vector<Record> records)
{

    vector<vector<P>> records_posit = convertPosit<P>(records);

    cout << setprecision(15);
    posit128_4 absErr = absError<P, T>(records_posit, records);
    cout << "Single vs Double absolute error: " << absErr << endl;

    posit128_4 relErr = relError<P, T>(records_posit, records);
    cout << "Single vs Double relative error: " << relErr << endl;

    double tol_test_pass = tol_test<P, T>(records_posit, records, 1e6);
    cout << "Single vs Double tolerance test pass rate: " << setprecision(3) << tol_test_pass << "%" << endl;

    // stringstream name;

    // name << "Generated"  << ".csv";
    // exportToCSV(records_posit, name.str());

    cout << endl;
    /*************/
    // Linear regression
    vector<P> x(records_posit.size());
    vector<P> y(records_posit.size());

    for (int i = 0; i < records_posit.size(); i++)
    {
        x[i] = records_posit[i][0];
        y[i] = records_posit[i][1];
    }

    double slope, intercept;
    // Perform linear regression
    linearRegression<P>(x, y, slope, intercept);
    cout << setprecision(10) << "slope: " << slope << ", "
         << "intercept: " << intercept << endl;

    double rSquared = RSquared(x, y, slope, intercept);
    cout << "R-squared: " << rSquared << endl;
}

int main(int argc, char **argv)
{
    if (argc < 4)
    {
        cout << "usage: " << argv[0] << " <filename> <Posit size> <Exponent bit size>\n";
        return 1;
    }

    string filename = argv[1];
    // int numCol_Ignore = stoi(argv[2]);
    vector<Record> records = parseCSV(filename);
    cout << "Parse completed: " << filename << endl;

    size_t totalData = records[0].getFieldCount() * records.size();
    cout << "The total number of data is: " << totalData << endl;

    // vector<vector<posit32_2>> records_posit32_2 = convertPosit<posit32_2>(records);

    /**
    for (const Record &record : records)
    {
        // Example: Print all fields
        for (size_t i = 0; i < record.getFieldCount(); ++i)
        {
            cout << record.getField<double>(i) << ", ";
        }
        cout << endl;
    }

    vector<vector<posit32_1>> record_posit = convertPosit<posit32_1>(records);

    for (const Record &record : records)
    {
        // Example: Print all fields
        for (size_t i = 0; i < record.getFieldCount(); ++i)
        {
            cout << record.getField<float>(i) << ", ";
        }
        cout << endl;
    }**/

    cout << endl;

    unsigned int p_size = stoi(argv[2]);
    unsigned int p_es = stoi(argv[3]);

    posit128_4 absErr;
    posit128_4 relErr;

    if (p_size == 16)
    {
        if (p_es == 0)
        {
            cout << properties<16, 0>();
            printResultPosit<posit16_0, double>(records, filename, p_size, p_es);
        }

        else if (p_es == 1)
        {
            cout << properties<16, 1>();
            printResultPosit<posit16_1, double>(records, filename, p_size, p_es);
        }
        else if (p_es == 2)
        {
            cout << properties<16, 2>();
            printResultPosit<posit16_2, double>(records, filename, p_size, p_es);
        }
        else if (p_es == 3)
        {
            cout << properties<16, 3>();
            printResultPosit<posit16_3, double>(records, filename, p_size, p_es);
        }

        else if (p_es == 4)
        {
            cout << properties<16, 4>();
            printResultPosit<posit16_4, double>(records, filename, p_size, p_es);
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
            printResultPosit<posit32_0, double>(records, filename, p_size, p_es);
        }
        else if (p_es == 1)
        {
            cout << properties<32, 1>();
            printResultPosit<posit32_1, double>(records, filename, p_size, p_es);
        }
        else if (p_es == 2)
        {
            cout << properties<32, 2>();
            printResultPosit<posit32_2, double>(records, filename, p_size, p_es);
        }
        else if (p_es == 3)
        {
            cout << properties<32, 3>();
            printResultPosit<posit32_3, double>(records, filename, p_size, p_es);
        }
        else if (p_es == 4)
        {
            cout << properties<32, 4>();
            printResultPosit<posit32_4, double>(records, filename, p_size, p_es);
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

    printResultIEEE<float, double>(records);
    printResultIEEE<double, double>(records);

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
