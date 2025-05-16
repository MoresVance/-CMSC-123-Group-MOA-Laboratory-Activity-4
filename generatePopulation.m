function pops = generatePopulation(num_pops, input_layer, hidden_layer, num_labels)
%GENERATEPOPULATION generates a variable number of populations (dependent on num_pops)
%and outputs it as cell array of weight matrices


pops = cell(num_pops,1);

for i = 1:num_pops
  %Random generation of weights
  Theta1 = randInitializeWeights(input_layer, hidden_layer); %Theta1 are the weights between input layer and hidden layer
  Theta2 = randInitializeWeights(hidden_layer, num_labels); %Theta2 are the weights between hidden layer and output layer
  nn_params = [Theta1(:); Theta2(:)]; % combine them into one (for use later for the cost function)
  pops(i) = nn_params; % full pops and stuff ye
end

end
