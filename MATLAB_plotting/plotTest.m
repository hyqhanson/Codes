
clc
close all
% Posit precision
Posit_16 = ["16,0" "16,1" "16,2" "16,3" "16,4"];
Posit_32 = [ "32,0" "32,1" "32,2" "32,3" "32,4"];
Posit_32_rmfirst = [ "32,1" "32,2" "32,3" "32,4"];
Posit_32_rmtwo = [ "32,2" "32,3" "32,4"];

%% Credit card
% Absolute error
creditcard_32_A = [2.77435745641396e-07, 1.74253482041884e-08,8.11404690985849e-09,9.01776972896746e-09,1.30317372420151e-08];
creditcard_F_A = [5.23179225787886e-08 ,5.23179225787886e-08,5.23179225787886e-08,5.23179225787886e-08,5.23179225787886e-08 ];
%dualBarPlot(creditcard_32_A, creditcard_F_A, Posit_32,'Absolute Error(Posit vs. double)','Posit','IEEE single','r','y');

% Relative error
creditcard_16 = [0.025350463422015, 0.000406714946841983,0.00013706364826366,0.00019127328284244,0.000359079601022329];
creditcard_32 = [3.89277481139952e-09, 1.1773505869468e-09, 1.46389530694943e-09, 2.59536240600224e-09,5.13960029607833e-09];
creditcard_F = [2.06303826985062e-08,2.06303826985062e-08,2.06303826985062e-08,2.06303826985062e-08,2.06303826985062e-08];
%dualBarPlot(creditcard_32, creditcard_F, Posit_32,'Relative Error(Posit vs. double)','Posit','IEEE single','r','y');


%BarPlot(creditcard_16,Posit_16,'Relative Error(float vs. single)','Posit');

% Threshold
creditcard_32_T = [98.3,98.66,98.66,98.24,96.4]
creditcard_F_T = [76.94,76.94,76.94,76.94,76.94]
%dualBarPlot(creditcard_32_T, creditcard_F_T, Posit_32,'Threshold test pass rate in percentage(Posit vs. single)','Posit','IEEE single','r','g');

%% Stock
% Absolute error
stock_32_A = [62.0802422092928,7.81562330240392,3.91926635265961];
stock_F_A = [4.40237282436555,4.40237282436555,4.40237282436555];
%dualBarPlot(stock_32_A, stock_F_A, Posit_32_rmtwo,'Absolute Error(Posit vs. double)','Posit','IEEE single','r','y');

% Complete absolute error
stock_32_AFull = [167421680.364107,12557.6403574418,62.0802422092928,7.81562330240392,3.91926635265961];
stock_F_AFull = [4.40237282436555,4.40237282436555,4.40237282436555,4.40237282436555,4.40237282436555];
%dualBarPlot(stock_32_AFull, stock_F_AFull, Posit_32,'Absolute Error(Posit vs. double)','Posit','IEEE single','r','y');

% Relative error
stock_32 = [1.07770146299153e-08,1.92060502195353e-09,1.27843895546782e-09];
stock_F = [3.54198136596674e-09 ,3.54198136596674e-09 ,3.54198136596674e-09 ];
%dualBarPlot(stock_32, stock_F, Posit_32_rmtwo,'Relative Error(Posit vs. double)','Posit','IEEE single','r','y');

% Complete relative error
stock_32Full = [0.0192032773277003,1.30841740980045e-06,1.07770146299153e-08,1.92060502195353e-09,1.27843895546782e-09];
stock_FFull = [3.54198136596674e-09,3.54198136596674e-09,3.54198136596674e-09 ,3.54198136596674e-09 ,3.54198136596674e-09 ];
%dualBarPlot(stock_32Full, stock_FFull, Posit_32,'Relative Error(Posit vs. double)','Posit','IEEE single','r','y');


% Threshold
stock_32_T = [12.05,14,15.56,16.89,17.57];
stock_F_T = [16.79,16.79,16.79,16.79,16.79];
%dualBarPlot(stock_32_T, stock_F_T, Posit_32,'Threshold test pass rate in percentage(Posit vs. single)','Posit','IEEE single','r','g');


%% Marketing
% Absolute error
market_32_A = [3.05250390719047e-06,4.69882155083581e-07, 2.63047601929415e-07, 2.6596870950356e-07, 3.78016059431887e-07];
market_F_A = [1.41564260180402e-06,1.41564260180402e-06,1.41564260180402e-06,1.41564260180402e-06,1.41564260180402e-06];
dualBarPlot(market_32_A, market_F_A, Posit_32,'Absolute Error(Posit vs. double)','Posit','IEEE single','r','y');

% Relative error
market_32 = [1.65096722229629e-08,3.56438352439219e-09,2.64885193178248e-09,2.95083694287703e-09,5.64642160920139e-09];
market_F = [2.11066370269214e-08,2.11066370269214e-08,2.11066370269214e-08,2.11066370269214e-08,2.11066370269214e-08];
dualBarPlot(market_32, market_F, Posit_32,'Relative Error(Posit vs. double)','Posit','IEEE single','r','y');

%Threshold
market_32_T = [21.05,32.02,32.02,23.68,11.99];
market_F_T = [4.386,4.386,4.386,4.386,4.386];
dualBarPlot(market_32_T, market_F_T, Posit_32,'Threshold test pass rate in percentage(Posit vs. single)','Posit','IEEE single','r','g');

%% Wine quality
% Absolute error
wine_32_A = [4.06587987379681e-09,2.41003857711387e-09,2.7942563630106e-09,5.67078995676246e-09,1.12309060703215e-09];
wine_F_A = [4.506483824993e-08,4.506483824993e-08,4.506483824993e-08,4.506483824993e-08,4.506483824993e-08];
%dualBarPlot(wine_32_A, wine_F_A, Posit_32,'Absolute Error(Posit vs. double)','Posit','IEEE single','r','y');

% Relative error
wine_32 = [1.01529299646738e-09, 6.70546980832827e-10,9.47411200555831e-10 ,1.89366687822361e-09,3.76529419326975e-09];
wine_F = [1.53103337875035e-08,1.53103337875035e-08,1.53103337875035e-08,1.53103337875035e-08,1.53103337875035e-08];
%dualBarPlot(wine_32, wine_F, Posit_32,'Relative Error(Posit vs. double)','Posit','IEEE single','r','y');

%Threshold
wine_32_T = [90.77,95.5,95.5,88.76,82.81];
wine_F_T = [67.77,67.77,67.77,67.77,67.77];
%dualBarPlot(wine_32_T, wine_F_T, Posit_32,'Threshold test pass rate in percentage(Posit vs. single)','Posit','IEEE single','r','g');


