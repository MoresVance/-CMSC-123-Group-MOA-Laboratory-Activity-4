% Load data
data = dlmread('cancer_training.data', ',', 0, 0);  % Read only numeric data

% Extract features (columns 2 to 10, excluding the ID and classification on the 11th column)
X = data(:, 2:10); % This is the training data

% Read labels separately (last column)
y = data(:, 11);

% Update class labels for the cancer dataset
class_labels = [1, 2]; % Assuming 1 = Benign, 2 = Malignant
if ~all(ismember(unique(y), class_labels))
    error('Unexpected class labels in the dataset. Check the data.');
end

% Some Constants
input_layer = 9; % 9 features
hidden_layer = 120; % arbitrary amount
num_labels = 2; % 2 classifications, Benign and Malignant

% PSO Parameters
MAX_ITERATIONS = 500; % Maximum iterations for PSO
SWARM_SIZE = 80; % Total number of particles
W = 0.90; % Inertia weight
C1 = 1.5; % Cognitive coefficient
C2 = 1.5; % Social coefficient

% Define the range of the search space for particle positions
position_min = -1; % Minimum value for particle positions
position_max = 1;  % Maximum value for particle positions

% Generate initial population of particles
particles = generatePopulation(SWARM_SIZE, input_layer, hidden_layer, num_labels);

% Initialize velocities for each particle
velocity = cell(SWARM_SIZE, 1);
for i = 1:SWARM_SIZE
    velocity{i} = zeros(size(particles{i})); % Initialize velocity to zero
end

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
        V_max = 0.2 * (position_max - position_min);
        velocity{i} = max(min(velocity{i}, V_max), -V_max);
        velocity{i} = W * velocity{i} + ...
                      C1 * rand() * (personal_best{i} - particles{i}) + ...
                      C2 * rand() * (global_best - particles{i});
        particles{i} = particles{i} + velocity{i};
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
data = dlmread('cancer_testing.data', ',', 0, 0);  % Read only numeric data
testing_data = data(:, 2:10); % This is the testing data
testing_labels = data(:, 11);

% Make predictions
result = predict(Theta1, Theta2, testing_data);
training_acc = mean(double(result == testing_labels)) * 100;
fprintf('Training Accuracy: %.2f%%\n', training_acc);

% Save parameters and accuracy to log.txt
log_file = fopen('log.txt', 'a'); % Open log.txt in append mode
if log_file == -1
    error('Could not open log.txt for writing.');
end

fprintf(log_file, "Run Date: %s\n", datestr(now));
fprintf(log_file, "PSO Parameters:\n");
fprintf(log_file, "  MAX_ITERATIONS: %d\n", MAX_ITERATIONS);
fprintf(log_file, "  SWARM_SIZE: %d\n", SWARM_SIZE);
fprintf(log_file, "  W: %.2f\n", W);
fprintf(log_file, "  C1: %.2f\n", C1);
fprintf(log_file, "  C2: %.2f\n", C2);
fprintf(log_file, "Training Accuracy: %.2f%%\n", training_acc);
fprintf(log_file, "----------------------------------------\n");

fclose(log_file); % Close the file