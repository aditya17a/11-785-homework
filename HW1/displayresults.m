function displayresults(X,Y,n)
figure()
points = Y(Y==1);
%plot(X(points,1),X(points,2),'*');
for i=1:length(Y)
    if Y(i)==1
        plot(X(i,1), X(i,2),'xr');
        hold on;
    end
end

%filename = strcat('RShape_',num2str(n),'nodes_output');
%fig = plot(X(points,:), 'r*');
%saveas(fig,filename,'jpg');
end

