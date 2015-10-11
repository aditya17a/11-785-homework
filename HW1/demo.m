seq = -1:0.005:1;
m = length(seq);
[xx,yy] = meshgrid(seq,seq);
X = [reshape(xx,1,numel(xx));reshape(yy,1,numel(yy))]';
[predtest, activations_final] = predictMultilayer(final_weights,X,1,2,...
    8);
activations = cell(size(activations_final));

for i = 1:length(activations_final)
activations{i} = activations_final{i}(:,2:end);
end

act2 = activations{2};
act3 = activations{3};
act4 = activations{4};

% for i=1:size(act3,2)
%     matrix = reshape(act2(:,i),m,[]);
%     %subplot(q/2,2,i);
%     figure(i);
%     filename = strcat('DRShape_hoddenLayer2_node_',num2str(i));
%     fig=imagesc([-1 1],[-1 1],matrix,[0 1]);
%     saveas(fig,filename,'jpg')
%     hold on;   
% end

matrix1 = reshape(activations_final{4},m,[]);
figure();
filename1 = strcat('DRShape_output_node');
fig1=imagesc([-1 1],[-1 1],matrix1,[0 1]);
saveas(fig1,filename1,'jpg')
hold on;   