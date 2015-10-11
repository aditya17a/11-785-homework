function Theta = RandomWeights(IncomingLayerSize, OutgoingLayerSize)
%Written by Aditya Sharma, Carnegie Mellon University
%This function takes in the size of 2 consecutive layers of a neural
%network and generates a random weight matrix with all values between
%[-espilon, +epsilon].

epsilon = sqrt(6)/sqrt(IncomingLayerSize +OutgoingLayerSize);
Theta = rand(OutgoingLayerSize, IncomingLayerSize+1)*(2*epsilon) - epsilon;
end