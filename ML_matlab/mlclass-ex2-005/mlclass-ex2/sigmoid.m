function g = sigmoid(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).


g = 1 ./ (1 + exp(-z));


% =============================================================

end

% sigmoid(0) = 0.5000
% sigmoid(5) = 0.9933.........a constant
% b = [1 2 3 4] ... a vector
% sigmoid(b) = [0.7311    0.8808    0.9526    0.9820]
% A = [1 2; 3 4] ......a matrix
% sigmoid(A) = [0.7311    0.8808; 0.9526    0.9820]
