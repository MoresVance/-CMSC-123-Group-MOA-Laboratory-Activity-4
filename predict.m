function predictions = predict(Theta1, Theta2, X)
    % Add bias unit to the input layer
    m = size(X, 1);
    X = [ones(m, 1) X]; % Add bias term

    % Forward propagation
    z2 = X * Theta1';
    a2 = sigmoid(z2);
    a2 = [ones(m, 1) a2]; % Add bias term to hidden layer

    z3 = a2 * Theta2';
    a3 = sigmoid(z3); % Output layer

    % Get predicted class labels
    [~, predictions] = max(a3, [], 2);
end

function g = sigmoid(z)
    g = 1 ./ (1 + exp(-z));
end