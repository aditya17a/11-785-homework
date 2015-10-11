function [J, grad] = CostFunction(X, Y, UnrolledWeights, InputLayerSize, HiddenLayerSize,...
    OutputLayerSize, lambda)
% Written by Aditya Sharma, Carnegie Mellon University
% This function implements forward and backpropogaiton to output the cost
% function J and the gradient of a neural network with input data X, output
% data Y, initialized and 'unrolled into a vector' weights. The function
% also takes the layer sizes and regularization parameter lambda

Theta1 = reshape(UnrolledWeights(1:HiddenLayerSize*(InputLayerSize+1)), ...
    HiddenLayerSize,(InputLayerSize+1)); %Size = HiddenLayer x (IpLayer+1)
Theta2 = reshape(UnrolledWeights(HiddenLayerSize*(InputLayerSize+1)+1:end), ...
    OutputLayerSize, (HiddenLayerSize+1)); %Size = OpLayer x (HiddenLayer+1)
%Theta1
%Theta2

%Feedforward step. 
M = size(X,1); % Number of datapoints
a1 = [ones(M,1) X]; % Size of a1 = M x(InputLayerSize + 1)
z2 = a1*Theta1'; % Size of z2 = M x HiddenLayerSize
a2 = [ones(M,1) sigmoid(z2)]; % Size of z2 = M x (HiddenLayerSize + 1)
z3 = a2*Theta2'; %Size of z3 = M x OutputLayerSize
a3 = sigmoid(z3);

%Computing the cost function J
J=0;
logterm = -Y.*log(a3)-(1-Y).*(log(1-a3));
Theta1n=Theta1(:,2:end);
Theta2n=Theta2(:,2:end);
J=((1/M).*sum(sum(logterm)))+(lambda/(2*M)).*(sum(sum(Theta1n.^2))+sum(sum(Theta2n.^2)));

%Backpropogation step
delta3 = a3-Y; %Size of delta3 = M x OutputLayerSize
delta2 = (delta3*Theta2).*SigmoidGradient([ones(M,1) z2]);
delta2 = delta2(:,2:end);

bigdelta1 = 0;
bigdelta2 = 0;
bigdelta1 = bigdelta1 + delta2'*a1;
bigdelta2 = bigdelta2 + delta3'*a2;
gradTheta1 = (1/M).*bigdelta1;
gradTheta2 = (1/M).*bigdelta2;

grad = [gradTheta1(:); gradTheta2(:)];

end