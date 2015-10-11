clear all;
close all;
load('figs.mat');

x2=Circle.test_x(:,2);
x1=Circle.test_x(:,1);
y=Circle.test_y;
figure();
for i=1:1:length(y)/10
    if y(i)==1
        plot(x1(i),x2(i),'*')
        hold on
    end
end
