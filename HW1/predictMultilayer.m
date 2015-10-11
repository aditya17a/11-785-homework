function [predictions ,activations] = predictMultilayer(UnrolledWeights,...
    X, numClasses, num_hidden_layers,size_hidden_layer)
%% ==================Initial Declarations and Assignments==================

M = size(X,1);
size_input_layer = size(X,2);
size_output_layer = numClasses;
total_layers = 2+num_hidden_layers;
numThetas = num_hidden_layers + 1;

%% =======================Unrolling Weights================================

Thetas = cell(numThetas,1);

Thetas{1} = reshape(UnrolledWeights(1:size_hidden_layer*(size_input_layer+1)), ...
    size_hidden_layer,(size_input_layer+1)); %Size = HiddenLayer x (IpLayer+1)
number = numel(Thetas{1});
top_pointer = number+1;
if numThetas > 2
    for i = 2:1:(numThetas-1)        
        Thetas{i} = reshape(UnrolledWeights(top_pointer:...
            top_pointer-1+(size_hidden_layer*(size_hidden_layer+1))), ...
    size_hidden_layer, (size_hidden_layer+1));

    number = number + numel(Thetas{i});
    top_pointer = number+1;
    end
end
Thetas{numThetas} = reshape(UnrolledWeights(top_pointer:end), ...
    size_output_layer, (size_hidden_layer+1)); %Size = OpLayer x (HiddenLayer+1)


%% ========================= Feedforward Step =============================

a1 = [ones(M,1) X]; % Size of a1 = M x(InputLayerSize + 1)
a=cell(total_layers,1);
z=cell(total_layers,1);
a{1} = a1;
z{1} = a1;

for i=2:1:total_layers-1
    z{i} = a{i-1}*Thetas{i-1}';
    a{i} = [ones(M,1) sigmoid(z{i})]; 
end
z{total_layers} = a{total_layers-1}*Thetas{total_layers-1}';
a{total_layers} = sigmoid(z{total_layers});

activations =a;

%% ===================== Generate predicitons  ============================
predictions = a{total_layers};
predictions(predictions<0.5)=0;
predictions(predictions>=0.5)=1;

end