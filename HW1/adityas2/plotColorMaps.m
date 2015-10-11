function plotColorMaps(Theta1, Theta2, n)
%Written by Aditya Sharma. This function plots the hidden layer activation
%on a color map. The inputs needed are the origin(x,y), corner(x,y) and the
%step size(x,y) that define the data space. The values of the weigths of
%the trained network are also needed.

seq = -1:0.005:1;
m = length(seq);
[xx,yy] = meshgrid(seq,seq);
X = [reshape(xx,1,numel(xx));reshape(yy,1,numel(yy))]';

[~,activations, activation_output] = predict(Theta1,Theta2,X);
activations = activations(:,2:end);
%figure();

q = size(activations,2); % Number of hidden neurons

for i=1:size(activations,2)
    matrix = reshape(activations(:,i),m,[]);
    %subplot(q/2,2,i);
    figure(i);
    filename = strcat('RShape_',num2str(n),'nodes_node_',num2str(i));
    fig=imagesc([-1 1],[-1 1],matrix,[0 1]);
    saveas(fig,filename,'jpg')
    hold on;   
end

matrix1 = reshape(activation_output,m,[]);
figure(q+1);
filename1 = strcat('RShape_',num2str(n),'output_node');
fig1=imagesc([-1 1],[-1 1],matrix1,[0 1]);
saveas(fig1,filename1,'jpg')
hold on;   
    

end