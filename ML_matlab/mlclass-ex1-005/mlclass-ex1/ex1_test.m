% setting up parameters for lin Reg in 1 var 
%           htheta(x) = theta'*X
%We add another dimension to our data X to accommodate the ?0 intercept term.
% We also initialize the initial parameters to 0 and the learning rate alpha
% to 0.01.
%Add a column of ones to x 
X = [ones(m, 1), data(:,1)];
%initializing fitting parameters
theta = zeros(2,1);
y = data(:,2);
iterations = 1500;
alpha = 0.01;

J = computeCost(X, y, theta);
J % 32.07

% run gradient descent
theta = gradientDescent(X, y, theta, alpha, iterations);


% print theta to screen
fprintf('Theta found by gradient descent: ');
fprintf('%f %f \n', theta(1), theta(2));
%output: Theta found by gradient descent: -3.630291 1.166362 

predict1 = [1, 3.5] * theta;
predict2 = [1, 7] * theta;

%Your final values for ? will also be used to make predictions on profits 
%in areas of 35,000(3.5*10,000) and 70,000(7*10,000) people.
fprintf('predict1 = %f,  predict2 = %f \n', predict1, predict2);
% output: predict1 = 0.451977,  predict2 = 4.534245 
% profits in areas of 35,000 people is $4,519.77 and profits in areas of 
% 70,000 people is $45,342.45

figure; % open a new figure window
plot(X(:,2),y,'rx','MarkerSize',10)
hold on
y1 = theta(1)+theta(2)*X(:,2);
plot(X(:,2),y1,'b');

xlabel("Population of City in 10,000s")
ylabel("Profit in $10,000s")
title("Population VS Revenue")
legend('Training data','Linear Regression');
hold off

costFunctionPlot();