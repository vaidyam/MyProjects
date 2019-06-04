data = load('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% Print out some data points
fprintf('First 10 examples from the dataset: \n');
fprintf(' x = [%.0f %.0f], y = %.0f \n', [X(1:10,:) y(1:10,:)]');
%size(X) . ..... 47 x 2
%size(y)........ 47 x 1

fprintf('Normalizing Features ...\n');
[X mu sigma] = featureNormalize(X);

X = [ones(m, 1) X];

fprintf('Running gradient descent ...\n');

% Choose some alpha value
alpha = 0.01;
num_iters = 400;

theta = zeros(3,1);

[theta1, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);

[theta2, J_history2] = gradientDescentMulti(X, y, theta, alpha*3, num_iters);
[theta3, J_history3] = gradientDescentMulti(X, y, theta, 0.1, num_iters);
[theta4, J_history4] = gradientDescentMulti(X, y, theta, 0.3, num_iters);

% Plot the convergence graph
figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');

figure
plot(1:50, J_history(1:50),'bo');
hold on;
plot(1:50, J_history2(1:50),'rx');
hold on
plot(1:50, J_history3(1:50), 'k+');
hold on
plot(1:50, J_history4(1:50), 'g*');
legend('learning_rate 0.01', 'learning_rate 0.03', 'learning_rate 0.1', 'learning_rate 0.3');
hold off

% Display gradient descent's result
fprintf('Theta computed from gradient descent: \n');
fprintf(' %f \n', theta1);
fprintf('\n');

% Estimate the price of a 1650 sq-ft, 3 br house
% ====================== YOUR CODE HERE ======================
% Recall that the first column of X is all-ones. Thus, it does
% not need to be normalized.
price = [1 1650 3] * theta1; % You should change this


fprintf('For a 1650 sq-ft and a 3 br house, we predict the price as %f\n',...
    price); % $166114823.98 ~~329900 <- [1600 3]

% ============================================================

fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using gradient descent):\n $%f\n'], price);

     
data = csvread('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% Add intercept term to X
X = [ones(m, 1) X];

% Calculate the parameters from the normal equation
theta = normalEqn(X, y);

% Display normal equation's result
fprintf('Theta computed from the normal equations: \n');
fprintf(' %f \n', theta);
fprintf('\n');


% Estimate the price of a 1650 sq-ft, 3 br house
% ====================== YOUR CODE HERE ======================
price = [1 1650 3] * theta; % You should change this


% ============================================================

fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using normal equations):\n $%f\n'], price); % $293081.46

     