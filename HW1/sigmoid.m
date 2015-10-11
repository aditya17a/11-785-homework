function f = sigmoid(x)
% Written by Aditya Sharma. This function takes an input vector or matrix x
% and outputs a sigmoid.
f = 1.0 ./ (1.0 + exp(-x));
end