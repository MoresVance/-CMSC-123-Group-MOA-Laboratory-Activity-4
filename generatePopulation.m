function population = generatePopulation(total_population, input_layer, hidden_layer, num_labels)
    population = cell(total_population, 1);
    for i = 1:total_population
        % Initialize weights for the ANN
        Theta1 = rand(hidden_layer, input_layer + 1) * 2 - 1; % Random weights for layer 1
        Theta2 = rand(num_labels, hidden_layer + 1) * 2 - 1; % Random weights for layer 2
        
        % Flatten the weights into a single vector
        population{i} = [Theta1(:); Theta2(:)];
    end
end