function [J, grad] = nnCostFunction(nn_params, input_layer, hidden_layer, num_labels, X, y, lambda)
    Theta1 = reshape(nn_params(1:hidden_layer * (input_layer + 1)), ...
                     hidden_layer, (input_layer + 1));
    Theta2 = reshape(nn_params((1 + (hidden_layer * (input_layer + 1))):end), ...
                     num_labels, (hidden_layer + 1));

    m = size(X, 1);
    J = 0;
    Theta1_grad = zeros(size(Theta1));
    Theta2_grad = zeros(size(Theta2));

    % Forward propagation
    a1 = [ones(m, 1) X]; % Add bias unit
    z2 = a1 * Theta1';
    a2 = [ones(size(z2, 1), 1) sigmoid(z2)]; % Add bias unit
    z3 = a2 * Theta2';
    a3 = sigmoid(z3); % Output layer

    Y = eye(num_labels)(y, :);


    % Cost function
    J = (1/m) * sum(sum(-Y .* log(a3) - (1 - Y) .* log(1 - a3))) + ...
        (lambda/(2*m)) * (sum(sum(Theta1(:, 2:end).^2)) + sum(sum(Theta2(:, 2:end).^2)));

    % Backpropagation
    delta3 = a3 - Y;
    delta2 = (delta3 * Theta2(:, 2:end)) .* sigmoidGradient(z2);

    Theta1_grad = (1/m) * (delta2' * a1);
    Theta2_grad = (1/m) * (delta3' * a2);

    % Regularization
    Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + (lambda/m) * Theta1(:, 2:end);
    Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + (lambda/m) * Theta2(:, 2:end);

    % Unroll gradients
    grad = [Theta1_grad(:); Theta2_grad(:)];
end

function g = sigmoid(z)
    g = 1.0 ./ (1.0 + exp(-z));
end

function g = sigmoidGradient(z)
    g = sigmoid(z) .* (1 - sigmoid(z));
end