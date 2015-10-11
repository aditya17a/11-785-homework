clc; clear all; close all;

%Loading the data files
load('figs.mat');
X = RShape.train_x;
Y = RShape.train_y;
test_x = RShape.test_x;
test_y = RShape.test_y;

%Declaring network size
size_input_layer = size(X,2);
size_output_layer = size(Y,2);
size_hidden_layer = 5;

%Initialize random weights and unroll them into a vector for use with cost
%function
Theta1 = RandomWeights(size_input_layer, size_hidden_layer);
Theta2 = RandomWeights(size_hidden_layer, size_output_layer);
initial_weights = [Theta1(:); Theta2(:)];
lambda = 1;

%Training Neural Network
fprintf('\nTraining Network... \n')
options = optimset('MaxIter', 500, 'MaxFunEvals', 13000);

ncostFunction = @(p) CostFunction(X, Y, p, size_input_layer, size_hidden_layer,...
    size_output_layer, lambda);

[final_weights, cost] = fmincg(ncostFunction, initial_weights, options);

Theta1 = reshape(final_weights(1:size_hidden_layer*(size_input_layer+1)), ...
    size_hidden_layer,(size_input_layer+1)); %Size = HiddenLayer x (IpLayer+1)

Theta2 = reshape(final_weights(size_hidden_layer*(size_input_layer+1)+1:end), ...
    size_output_layer, (size_hidden_layer+1)); %Size = OpLayer x (HiddenLayer+1)

%Make predictions
pred = predict(Theta1, Theta2, X);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == Y)) * 100);

[predtest,a2] = predict(Theta1, Theta2, test_x);
fprintf('\nTest Set Accuracy: %f\n', mean(double(predtest == test_y)) * 100);
% displayresults(X,pred);

% Generating color maps
% origin = [0.001,0.001];
% corner = [1,1];
% step = [0.0001,0.0001];

%plotColorMaps(Theta1,Theta2,size_hidden_layer);
%displayresults(test_x,test_y,size_hidden_layer);

