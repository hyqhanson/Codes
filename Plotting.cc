#include <iostream>
#include <fstream>
#include <vector>
#include "class.h"

using namespace std;

void plot(vector<double>& points){

    // Write points to a data file
    ofstream dataFile("points.dat");
    if (!dataFile.is_open()) {
        cerr << "Error: Failed to open data file." << endl;
        return;
    }

    for (size_t i = 0; i < points.size(); ++i) {
        // Write the x-coordinate and the value as the y-coordinate
        dataFile << i + 1 << " " << points[i] << endl;
    }
    dataFile.close();

    // Plot data file using Gnuplot
    FILE *pipe = popen("gnuplot -persist", "w");
    if (!pipe) {
        cerr << "Error: Couldn't open gnuplot pipe!" << endl;
        return;
    }

    // Plot points from data file
    fprintf(pipe, "plot 'points.dat' with points pointtype 7 pointsize 1, '' with lines\n");

    // Close the pipe
    pclose(pipe);

}
