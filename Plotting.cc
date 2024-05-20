#include <iostream>
#include <fstream>
#include <vector>
#include "class.h"

using namespace std;

template <typename P>
void plot(vector<P> &points)
{

    // Write points to a data file
    ofstream dataFile("points.dat");
    if (!dataFile.is_open())
    {
        cerr << "Error: Failed to open data file." << endl;
        return;
    }

    for (size_t i = 0; i < points.size(); ++i)
    {
        // Write the x-coordinate and the value as the y-coordinate
        dataFile << i + 1 << " " << points[i] << endl;
    }
    dataFile.close();

    // Plot data file using Gnuplot
    FILE *pipe = popen("gnuplot -persist", "w");
    if (!pipe)
    {
        cerr << "Error: Couldn't open gnuplot pipe!" << endl;
        return;
    }

    // Plot points from data file
    fprintf(pipe, "plot 'points.dat' with points pointtype 7 pointsize 1, '' with lines\n");

    // Close the pipe
    pclose(pipe);
}

template <typename P, typename T>
void plot(vector<P> &points1, vector<T> &points2)
{
    // Write points1 to a data file
    ofstream dataFile1("points1.dat");
    if (!dataFile1.is_open())
    {
        cerr << "Error: Failed to open data file for points1." << endl;
        return;
    }

    for (size_t i = 0; i < points1.size(); ++i)
    {
        // Write the x-coordinate and the value as the y-coordinate
        dataFile1 << i + 1 << " " << points1[i] << endl;
    }
    dataFile1.close();

    // Write points2 to a data file
    ofstream dataFile2("points2.dat");
    if (!dataFile2.is_open())
    {
        cerr << "Error: Failed to open data file for points2." << endl;
        return;
    }

    for (size_t i = 0; i < points2.size(); ++i)
    {
        // Write the x-coordinate and the value as the y-coordinate
        dataFile2 << i + 1 << " " << points2[i] << endl;
    }
    dataFile2.close();

    // Plot data files using Gnuplot
    FILE *pipe = popen("gnuplot -persist", "w");
    if (!pipe)
    {
        cerr << "Error: Couldn't open gnuplot pipe!" << endl;
        return;
    }

    // Plot points from both data files
    fprintf(pipe, "plot 'points1.dat' with points pointtype 7 pointsize 1 title 'Predicted Data', 'points2.dat' with points pointtype 7 pointsize 1 title 'Real Data'\n");

    // Close the pipe
    pclose(pipe);
}
