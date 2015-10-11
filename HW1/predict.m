function [p,a2,a3] = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)



M = size(X,1); % Number of datapoints
a1 = [ones(M,1) X]; % Size of a1 = M x(InputLayerSize + 1)
z2 = a1*Theta1'; % Size of z2 = M x HiddenLayerSize
a2 = [ones(M,1) sigmoid(z2)]; % Size of a2 = M x (HiddenLayerSize + 1)
z3 = a2*Theta2'; %Size of z3 = M x OutputLayerSize
a3 = sigmoid(z3);
p = a3;

p(p<0.5)=0;
p(p>=0.5)=1;
% =========================================================================


end