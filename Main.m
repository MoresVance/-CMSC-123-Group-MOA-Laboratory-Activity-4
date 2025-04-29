% Load data
data = dlmread('iris_training.data', ',', 0, 0);  % Read only numeric data

% Extract features (first 4 columns, excluding the classification on the 5th column)
X = data(:, 1:4); % This is the training data

% Read labels separately (since they are strings)
fid = fopen('iris_training.data', 'r');
if fid == -1
    error('File iris_training.data not found or cannot be opened.');
end


% Adjust delimiter if necessary
labels = textscan(fid, '%*f %*f %*f %*f %s', 'Delimiter', ','); % Update delimiter if needed
fclose(fid);

% Check if labels were read correctly
if isempty(labels{1})
    error('Labels could not be read. Check the file format and delimiter.');
end

y = labels{1};

class_labels = {'Iris-setosa', 'Iris-versicolor', 'Iris-virginica'};
[~, y] = ismember(y, class_labels); % this is the outcome for each of the training data

% Some Constants
input_layer = 4; % 4 features, sepal length & width, petal length & width
hidden_layer = 120; % arbitrary amount
num_labels = 3; % 3 classifications, Setosa, Versicolor, Virginica

% PSO Parameters
MAX_ITERATIONS = 100; % Maximum iterations for PSO
SWARM_SIZE = 50; % Total number of particles
W = 0.5; % Inertia weight
C1 = 1.5; % Cognitive coefficient
C2 = 1.5; % Social coefficient

% Generate initial population of particles
particles = generatePopulation(SWARM_SIZE, input_layer, hidden_layer, num_labels);

% Initialize personal best and global best
personal_best = particles;
personal_best_fitness = inf(SWARM_SIZE, 1);
global_best = [];
global_best_fitness = inf;

% PSO Training Loop
for iter = 1:MAX_ITERATIONS
    for i = 1:SWARM_SIZE
        nn_params = particles{i};
        fitness = nnCostFunction(nn_params, input_layer, hidden_layer, num_labels, X, y, 1);
        
        % Update personal best
        if fitness < personal_best_fitness(i)
            personal_best_fitness(i) = fitness;
            personal_best{i} = nn_params;
        end
        
        % Update global best
        if fitness < global_best_fitness
            global_best_fitness = fitness;
            global_best = nn_params;
        end
    end
    
    % Update particle velocities and positions
    for i = 1:SWARM_SIZE
        % Update velocity and position here (not shown for brevity)
        % This would typically involve using W, C1, C2, and the best positions
    end
    
    printf("Iteration %d Global Best Fitness: %f\n", iter, global_best_fitness);
    
    % Stop if global best fitness is already 0
    if global_best_fitness == 0
        break;
    end
end

% Reshape the optimal weights for the ANN
Theta1 = reshape(global_best(1:hidden_layer * (input_layer + 1)), ...
                 hidden_layer, (input_layer + 1));

Theta2 = reshape(global_best((1 + (hidden_layer * (input_layer + 1))):end), ...
                 num_labels, (hidden_layer + 1));

% Load testing data
data = dlmread('iris_testing.data', ',', 0, 0);  % Read only numeric data
testing_data = data(:, 1:4); % This is the testing data

fid = fopen('iris_testing.data', 'r');
labels = textscan(fid, '%*f %*f %*f %*f %s', 'Delimiter', ',');
fclose(fid);

testing_labels = labels{1};

[~, testing_labels] = ismember(testing_labels, class_labels); % this is the outcome for each of the testing data

% Make predictions
result = predict(Theta1, Theta2, testing_data);
disp("Predicted Results: ");
disp(result);
disp("Actual Results: ");
disp(testing_labels);
training_acc = mean(double(result == testing_labels(1:length(result), 1))) * 100;
fprintf('Training Accuracy: %.2f%%\n', training_acc);