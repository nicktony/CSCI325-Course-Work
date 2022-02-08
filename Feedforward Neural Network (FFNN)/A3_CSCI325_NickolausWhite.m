% % % % % % % % % % % % % % % % % % %
% Nickolaus White (CSCI325)
% % % % % % % % % % % % % % % % % % %

% Close command window, workspace, and all figure pop-ups
%--------------------------------------------------------------------
clc
clear all
close all

% Load in data (differs based on your file location)
%--------------------------------------------------------------------
heightweightinputs = readtable('heightweight_inputs.xlsx');

% Grab inputs from table, height and weight of individuals
input = [heightweightinputs.Height'; heightweightinputs.Weight'];
targets = heightweightinputs.BMI';

% Create feed forward network
%--------------------------------------------------------------------
net = feedforwardnet(10, 'trainscg'); %w/ specified algorithm
%net = feedforwardnet(10); %w/o specified algorithm

% Train data
%--------------------------------------------------------------------
[net,tr] = train(net,input,targets);

% Output the 10th entry calculated from the NN
%--------------------------------------------------------------------
output = net(input(:,10))

% Save file contents, end of program
%---------------------------------------------------------------
filename = 'A3_CSCI325_NickolausWhite.mat';
save(filename)




