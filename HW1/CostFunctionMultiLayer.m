function [J, grad] = CostFunctionMultiLayer(X, Y, UnrolledWeights,...
    InputLayerSize, HiddenLayerSize,...
    OutputLayerSize, numHiddenlayers, lambda)
% Written by Aditya Sharma, Carnegie Mellon University
% This function implements forward and backpropogaiton to output the cost
% function J and the gradient of a neural network with input data X, output
% data Y, initialized and 'unrolled into a vector' weights. The function
% also takes the layer sizes and regularization parameter lambda

numThetas = numHiddenlayers + 1;
weightsCell = cell(numThetas,1);
weightsCell{1} = reshape(UnrolledWeights(1:HiddenLayerSize*(InputLayerSize+1)), ...
    HiddenLayerSize,(InputLayerSize+1)); %Size = HiddenLayer x (IpLayer+1)
number = numel(weightsCell{1});
top_pointer = number+1;
if numThetas > 2
    for i = 2:1:(numThetas-1)
        
        weightsCell{i} = reshape(UnrolledWeights(top_pointer:...
            top_pointer-1+(HiddenLayerSize*(HiddenLayerSize+1))), ...
    HiddenLayerSize, (HiddenLayerSize+1));

    number = number + numel(weightsCell{i});
    top_pointer = number+1;
    end
end
weightsCell{numThetas} = reshape(UnrolledWeights(top_pointer:end), ...
    OutputLayerSize, (HiddenLayerSize+1)); %Size = OpLayer x (HiddenLayer+1)
%Theta1
%Theta2

%% =======================Feedforward step================================= 

M = size(X,1); % Number of datapoints
a1 = [ones(M,1) X]; % Size of a1 = M x(InputLayerSize + 1)

total_layers = 2+numHiddenlayers;
a=cell(total_layers,1);
z=cell(total_layers,1);
a{1} = a1;
z{1} = a1;

for i=2:1:total_layers-1
    z{i} = a{i-1}*weightsCell{i-1}';
    a{i} = [ones(M,1) sigmoid(z{i})]; 
end
z{total_layers} = a{total_layers-1}*weightsCell{total_layers-1}';
a{total_layers} = sigmoid(z{total_layers});

% z2 = a1*Theta1'; % Size of z2 = M x HiddenLayerSize
% a2 = [ones(M,1) sigmoid(z2)]; % Size of z2 = M x (HiddenLayerSize + 1)
% z3 = a2*Theta2'; %Size of z3 = M x OutputLayerSize
% a3 = sigmoid(z3);

%% ===================Computing the cost function J=======================
J=0;
logterm = -Y.*log(a{total_layers})-(1-Y).*(log(1-a{total_layers}));
weightn = cell(size(weightsCell));

for i=1:1:numel(weightsCell)
    weightn{i}=weightsCell{i}(:,2:end);
end

regularization_term=0;

for i=1:1:numel(weightn)
    inner_square = weightn{i}.^2;
    regularization_term = regularization_term + sum(sum(inner_square));
end
J=((1/M).*sum(sum(logterm)))+(lambda/(2*M)).*regularization_term;


% J=0;
% logterm = -Y.*log(a3)-(1-Y).*(log(1-a3));
% Theta1n=Theta1(:,2:end);
% Theta2n=Theta2(:,2:end);
% J=((1/M).*sum(sum(logterm)))+(lambda/(2*M)).*(sum(sum(Theta1n.^2))+sum(sum(Theta2n.^2)));

%% =====================Backpropogation step===============================    

%Calculating deltas
delta=cell(total_layers,1);
delta{total_layers} = a{total_layers}-Y; %Size of delta3 = M x OutputLayerSize
for i=total_layers-1:-1:2
    delta{i} = (delta{i+1}*weightsCell{i}).*...
        SigmoidGradient([ones(M,1) z{i}]);
    delta{i} = delta{i}(:,2:end);
end
% % delta3 = a3-Y;
% % delta2 = (delta3*Theta2).*SigmoidGradient([ones(M,1) z2]);
% % delta2 = delta2(:,2:end);

%Calculating big deltas
bigdelta = cell(numThetas,1);
gradTheta = cell(numThetas,1);
for i=1:1:numThetas
    bigdelta{i} = delta{i+1}'*a{i};
    gradTheta{i} = (1/M).*bigdelta{i};
end
% % bigdelta1 = 0;
% % bigdelta2 = 0;
% % bigdelta1 = bigdelta1 + delta2'*a1;
% % bigdelta2 = bigdelta2 + delta3'*a2;
% % gradTheta1 = (1/M).*bigdelta1;
% % gradTheta2 = (1/M).*bigdelta2;

grad=[];
for i=1:1:numThetas
    grad = [grad;gradTheta{i}(:)];
end
%grad = [gradTheta1(:); gradTheta2(:)];

end