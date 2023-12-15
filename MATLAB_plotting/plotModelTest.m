clc
close all
% Posit precision
Posit_16 = ["16,0" "16,1" "16,2" "16,3" "16,4"];
Posit_32 = [ "32,0" "32,1" "32,2" "32,3" "32,4"];

%% Marketing (Multiple Linear Regression)
% 1000 0.00002
marketing_32 = [0.8769024068,0.8769026411,0.8769026396,0.8769026398,0.8769026425];
marketing_F = [0.8769025859,0.8769025859,0.8769025859,0.8769025859,0.8769025859];
marketing_D = [0.8769026395,0.8769026395,0.8769026395,0.8769026395,0.8769026395];
%triBarPlot(marketing_32, marketing_F,marketing_D, Posit_32,'R-squred','Posit','IEEE single','IEEE double','r','y','g');

%% Wine quality (Logistic Regression)
% 100 0.000002
wine_32 = [68.33,68.12,68.12,68.12,68.12]
wine_F = [62.29,62.29,62.29,62.29,62.29]
wine_D = [67.71,67.71,67.71,67.71,67.71]
triBarPlot(wine_32,wine_F,wine_D, Posit_32,'Accuracy','Posit','IEEE single','IEEE double','r','y','g');
