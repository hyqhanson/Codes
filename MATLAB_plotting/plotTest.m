
clc
close all
% Posit precision
Posit_16 = ["16,0" "16,1" "16,2" "16,3" "16,4"];
Posit_32 = [ "32,0" "32,1" "32,2" "32,3" "32,4"];
Posit_32_rmfirst = [ "32,1" "32,2" "32,3" "32,4"];

%% Credit card
% Absolute error
creditcard_32_A = [2.421122271511e-08,1.00261819740443e-08,1.06006852797396e-08,1.46514138381437e-08];
creditcard_F_A = [5.9104605866733e-08 ,5.9104605866733e-08 ,5.9104605866733e-08 ,5.9104605866733e-08 ];
%dualBarPlot(creditcard_32_A, creditcard_F_A, Posit_32_rmfirst,'Absolute Error(Posit vs. double)','Posit','IEEE float');

% Relative error
creditcard_16 = [0.025350463422015, 0.000406714946841983,0.00013706364826366,0.00019127328284244,0.000359079601022329];
creditcard_32 = [9.07922383493392e-07, 1.16515361418846e-09, 1.45907558720162e-09, 2.59394158637185e-09,5.14790342196719e-09];
creditcard_F = [2.0623091015924e-08, 2.0623091015924e-08,2.0623091015924e-08,2.0623091015924e-08,2.0623091015924e-08];
%dualBarPlot(creditcard_32, creditcard_F, Posit_32,'Relative Error(Posit vs. double)','Posit','IEEE float');

creditcard_32_rmfirst = [1.16515361418846e-09, 1.45907558720162e-09, 2.59394158637185e-09,5.14790342196719e-09];
creditcard_F_rmfirst = [2.0623091015924e-08,2.0623091015924e-08,2.0623091015924e-08,2.0623091015924e-08];
%dualBarPlot(creditcard_32_rmfirst, creditcard_F_rmfirst, Posit_32_rmfirst,'Relative Error(Posit vs. double) after remove (32,0)','Posit','IEEE float');

%BarPlot(creditcard_16,Posit_16,'Relative Error(float vs. single)','Posit');

% Threshold
creditcard_32_T = [40.8,45.1,36.7,25.4,17.4]
creditcard_F_T = [8.77,8.77,8.77,8.77,8.77]
%dualBarPlot(creditcard_32_T, creditcard_F_T, Posit_32,'Threshold test pass rate in percentage(Posit vs. single)','Posit','IEEE float');

%% Stock
% Absolute error
stock_32_A = [12557.6403574418,62.0802422092928,7.81562330240392,3.91926635265961];
stock_F_A = [4.40237282436555,4.40237282436555,4.40237282436555,4.40237282436555];
%dualBarPlot(stock_32_A, stock_F_A, Posit_32_rmfirst,'Absolute Error(Posit vs. double)','Posit','IEEE float');

% Relative error
stock_16 = [0.175894137929491,0.0410298644107166,0.00122325999268679,0.000436201907626087,0.00039026624351787];
stock_32 = [1.30841740980045e-06,1.07770146299153e-08,1.92060502195353e-09,1.27843895546782e-09];
stock_F = [3.54198136596674e-09 ,3.54198136596674e-09 ,3.54198136596674e-09 ,3.54198136596674e-09 ];
%dualBarPlot(stock_32, stock_F, Posit_32_rmfirst,'Relative Error(Posit vs. double) after remove (32,0)','Posit','IEEE float');
%dualBarPlot([1.07770146299153e-08,1.92060502195353e-09,1.27843895546782e-09],  ...
    %[3.54198136596674e-09 ,3.54198136596674e-09 ,3.54198136596674e-09], ...
   %[ "32,2" "32,3" "32,4"],'Relative Error(Posit vs. double) after remove (32,0),(32,1)','Posit','IEEE float');

%BarPlot(stock_16,Posit_16,'Relative Error(float vs. single)','Posit');

% Threshold
stock_32_T = [6.12,7.77,9.32,10.7,11.5];
stock_F_T = [10.7,10.7,10.7,10.7,10.7];
%dualBarPlot(stock_32_T, stock_F_T, Posit_32,'Threshold test pass rate in percentage(Posit vs. single)','Posit','IEEE float');

%% Climate change
% Relative error
climate_32 = [2.22252408792183e-08,3.95307020141426e-09 ,2.6692526629046e-09,3.15011082274056e-09 ,5.32871214398824e-09];
climate_F = [2.13005449697598e-08,2.13005449697598e-08,2.13005449697598e-08,2.13005449697598e-08,2.13005449697598e-08];
%dualBarPlot(climate_32, climate_F, Posit_32,'Relative Error(Posit vs. double)','Posit','IEEE float');

%Threshold
climate_32_T = [8.3,7.15,5.15,3.08,1.83];
climate_F_T = [0.59,0.59,0.59,0.59,0.59];
%dualBarPlot(climate_32_T, climate_F_T, Posit_32,'Threshold test pass rate in percentage(Posit vs. single)','Posit','IEEE float');

%% Daily climate
% Relative error
daily_32 = [1.43889788099874e-08 ,1.61956065610526e-09 ,1.16601133684167e-09,1.49747425068388e-09,2.61394148263437e-09];
daily_F = [1.04454669073294e-08,1.04454669073294e-08,1.04454669073294e-08,1.04454669073294e-08,1.04454669073294e-08];
dualBarPlot(daily_32, daily_F, Posit_32,'Relative Error(Posit vs. double)','Posit','IEEE float');

%Threshold
daily_32_T = [54.6,54,54,53.9,53.9];
daily_F_T = [53.9,53.9,53.9,53.9,53.9];
dualBarPlot(daily_32_T, daily_F_T, Posit_32,'Threshold test pass rate in percentage(Posit vs. single)','Posit','IEEE float');

%% Linear regression
% Relative error
linear_32 = [6.19288198551825e-09,1.75272516827919e-09,1.2279041332349e-09,1.32921981792038e-09,2.71016142157805e-09];
linear_F = [1.05079379319833e-08,1.05079379319833e-08,1.05079379319833e-08,1.05079379319833e-08,1.05079379319833e-08];
dualBarPlot(linear_32, linear_F, Posit_32,'Relative Error(Posit vs. double)','Posit','IEEE float');
