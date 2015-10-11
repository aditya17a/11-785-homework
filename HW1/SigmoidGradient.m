function f = SigmoidGradient(x)
% Written by Aditya Sharma. This function takes an input vector or matrix x
% and outputs a sigmoid gradient.
f=sigmoid(x).*(1-sigmoid(x));