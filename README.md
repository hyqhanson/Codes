# Introduction
This repository aims to provide a flexible environment for testing Posit with sizes 16/32 bits and es values 0/1/2/3/4 against Single-precision or Double-precision.

# How to use it
Before running this pacakge, you have to download and install the Universal library https://github.com/stillwater-sc/universal 

To compile it, you have to run the Makefile by simply run `make clean` and then `make` in the terminal

To run the program and conduct tests, follow the guides below:
```
./main <filename> <Posit size> <Exponent bit size> <Regression/Classification> <number of iteration> <learning rate>

<filename> are usually in the format of TestingFiles/filename.csv"
<Posit size> can be selected between 16 or 32
<Exponent bit size> can be selected between 0,1,2 3 or 4 
<Regression/Classification> can be decided by using 0: DON'T DO any model, or 1: do regression, or 2: do binary classfication

Example: ./main TestingFiles/Marketing.csv 32 2 1 1000 0.00002
This will use the dataset 'Marketing_new.csv' with 32 bits Posit with 2 es value, it will conduct linear regression with 1000 iteration and 0.00002 learning rate.
```
The output will show information as below:
```
Parse completed: TestingFiles/Marketing.csv
The total number of data is: 684

nbits   es      minpos          maxpos          epsilon 
32      2       7.52316e-37     1.32923e+36     7.45058e-09

Posit vs Double absolute error: 2.63047601929415e-07
Posit vs Double relative error: 2.64885193178248e-09
Posit vs Double tolerance test pass rate: 32.02%
Posit format data has been successfully exported to Generated_files/Generated_Marketing_32_2.csv

Using multiple linear regression: 
Learned Parameters (Theta): 0.9925041907 0.05083409837 0.2127422839 0.0176769793 
R-squared: 0.8769026396

====================================
Single vs Double: 
Absolute error: 1.41564260180402e-06
Relative error: 2.11066370269214e-08
Tolerance test pass rate: 4.386%

Using multiple linear regression: 
Learned Parameters (Theta): 0.9925013781 0.05083410814 0.212742269 0.01767701097 
R-squared: 0.8769025859
====================================
Double vs Double: 
Absolute error: 0
Relative error: 0
Tolerance test pass rate: 100%

Using multiple linear regression: 
Learned Parameters (Theta): 0.9925041991 0.05083409797 0.2127422828 0.0176769806 
R-squared: 0.8769026395
```



# Auto testing datasets
For testing multiple dataset, you should firstly put dataset as csv file in `TestingFiles/`, then put their names in `Name.txt`.
```
\\Name.txt
creditcard.csv
Stock.csv
climate_change.csv
winequality.csv
Marketing.csv

```
Then simply run `./auto_run.sh`, it will automatically test all the datasets with various Posit types in the comparison between Posit and IEEE (without using models), and generates results in `output_table.txt`


# Generated Posit precision files
Every time you run main or use `auto_run.sh`, it will generate a corresponding Posit precision file in `Generated_files/` with a filename `Generated_'filename'_'totalbits'_'es'.csv`