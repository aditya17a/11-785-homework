clc; clear all; close all;

%% ===================== Loading the data files ===========================
load('figs.mat');
X = Diamod.train_x;
Y = Diamod.train_y;
test_x = Diamod.test_x;
test_y = Diamod.test_y;

%% ==================== Declaring network size ============================
size_input_layer = size(X,2);
size_output_layer = size(Y,2);
size_hidden_layer = 5;
num_hidden_layers = 1;
num_classes = size_output_layer;

%% ========= Initialize random weights and unroll them into a vector ======

% if num_hidden_layer == 1
% Theta1 = RandomWeights(size_input_layer, size_hidden_layer);
% Theta2 = RandomWeights(size_hidden_layer, size_output_layer);
% initial_weights = [Theta1(:); Theta2(:)];
% end
initial_weights = [];%This will be the unrolled vector
numThetas = num_hidden_layers + 1;
weightsCell = cell(numThetas,1);
weightsCell{1} = RandomWeights(size_input_layer, size_hidden_layer);
initial_weights = [initial_weights; weightsCell{1}(:)];
if numThetas > 2
    for i = 2:1:(numThetas-1)
        weightsCell{i} = RandomWeights(size_hidden_layer, size_hidden_layer);
        initial_weights = [initial_weights; weightsCell{i}(:)];
    end
end
weightsCell{numThetas} = RandomWeights(size_hidden_layer, size_output_layer);
initial_weights = [initial_weights; weightsCell{numThetas}(:)];



%% ====================== Train Neural Network ============================
lambda = 1;
fprintf('\nTraining Network... \n')
options = optimset('MaxIter', 500, 'MaxFunEvals', 13000);

ncostFunction = @(p) CostFunctionMultiLayer(X, Y, p, size_input_layer, size_hidden_layer,...
    size_output_layer, num_hidden_layers,  lambda);

[final_weights, cost] = fmincg(ncostFunction, initial_weights, options);

%% ===================== Send trained weights for prediction===============


% Theta1 = reshape(final_weights(1:size_hidden_layer*(size_input_layer+1)), ...
%     size_hidden_layer,(size_input_layer+1)); %Size = HiddenLayer x (IpLayer+1)
% 
% Theta2 = reshape(final_weights(size_hidden_layer*(size_input_layer+1)+1:end), ...
%     size_output_layer, (size_hidden_layer+1)); %Size = OpLayer x (HiddenLayer+1)
% 
[pred, ~] = predictMultilayer(final_weights,X,num_classes,num_hidden_layers,...
    size_hidden_layer);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == Y)) * 100);
[predtest, activations_final] = predictMultilayer(final_weights,test_x,num_classes,num_hidden_layers,...
    size_hidden_layer);
fprintf('\nTest Set Accuracy: %f\n', mean(double(predtest == test_y)) * 100);


% %Make predictions
% pred = predict(Theta1, Theta2, X);
% fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == Y)) * 100);
% 
% [predtest,a2] = predict(Theta1, Theta2, test_x);
% fprintf('\nTest Set Accuracy: %f\n', mean(double(predtest == test_y)) * 100);
% % displayresults(X,pred);
% 
% % Generating color maps
% % origin = [0.001,0.001];
% % corner = [1,1];
% % step = [0.0001,0.0001];
% 
% %plotColorMaps(Theta1,Theta2,size_hidden_layer);
% %displayresults(test_x,test_y,size_hidden_layer);
% 
