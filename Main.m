% Load data
data = dlmread('cancer_training.data', ',', 0, 0);  % Read only numeric data

% Extract features 
X = data(:, 2:10); %This is the training data
y = data(:, 11:11) %This is traning labels

% Some Constants
input_layer = 9 
hidden_layer = 20% arbitrary amount
num_labels = 2 % 2 classifications 2 or 4
c1 = 2.15; % cognitive coefficient
c2 = 2.15; % social coefficient
wmax = 0.85; % max inertia weight
wmin = 0.65; % min inertia weight


MAX_GENERATIONS = 500 % Maximum Generations to go through
TOTAL_POPULATION = 100 % Total Population

V_max = 0.45; % Max Velocity

% GENERATE POPULATION
pops_position = generatePopulation(TOTAL_POPULATION, input_layer, hidden_layer, num_labels);
pops_velocity= generatePopulation(TOTAL_POPULATION, input_layer, hidden_layer, num_labels);
best_personal = pops_position;
best_global = pops_position{1};

optimal_weights = [];
minFitness = ones(MAX_GENERATIONS, 1);

fid1 = fopen('results.txt', 'w');

% Inital fitness evaluation
for i = 1: TOTAL_POPULATION
    nn_params = pops_position{i};
    best_personal_fitness = nnCostFunction(best_personal{i}, input_layer, hidden_layer, num_labels, X, y, 1);
    best_global_fitness = nnCostFunction(best_global, input_layer, hidden_layer, num_labels, X, y, 1);

    if (best_personal_fitness < best_global_fitness)
        best_global = best_personal{i};
    endif

end

%disp(pops_velocity);
%disp(new_velocity(pops_velocity, pops_position, best_personal, best_global, 1, 2, 2))

current_fitness = nnCostFunction(pops_position{1} + pops_velocity{1}, input_layer, hidden_layer, num_labels, X, y, 1);
% disp(current_fitness);
% FITNESS EVALUATION
for g = 1: MAX_GENERATIONS
    pops_position = updatePosition(pops_position, pops_velocity);
    %disp(pops_velocity{1});
    %disp(pops_position{1});
    for i = 1: TOTAL_POPULATION
        current_fitness = nnCostFunction(pops_position{i}, input_layer, hidden_layer, num_labels, X, y, 1);
        best_personal_fitness = nnCostFunction(best_personal{i}, input_layer, hidden_layer, num_labels, X, y, 1);
        global_fitness = nnCostFunction(best_global, input_layer, hidden_layer, num_labels, X, y, 1);

        if (current_fitness < best_personal_fitness)
            best_personal{i} = pops_position{i};
        endif 

        if (current_fitness < global_fitness)
            best_global = pops_position{i};
        endif 
    end

    pops_velocity = updateVelocity(pops_velocity, pops_position, best_personal, best_global, c1, c2, g, MAX_GENERATIONS, wmin, wmax, best_personal_fitness, global_fitness);
    
    for i = 1:TOTAL_POPULATION
        pops_velocity{i} = max(min(pops_velocity{i}, V_max), -V_max);
    end

    global_fitness = nnCostFunction(best_global, input_layer, hidden_layer, num_labels, X, y, 1);

    minFitness(g) = global_fitness;
    printf("Generation %d Global Fitness: %d\n", g, global_fitness);

    % sort the population according to fitness, in ascending order
    % [sorted_fitness, sorted_indices] = sort(fitness, 'ascend');
    % pops = pops(sorted_indices);

    % get the current minimum and set that as the optimal weight
    % optimal_weights = pops{1};
    % minFitness(g) = fitness(1);

    % printf("Generation %d Min Fitness: %d\n", g, fitness(1))

    % stop if fitness at that minimum is already 0
    % if (fitness(1) == 0)
    %     break;
    % endif

end

% xlabel ("Generation");
% p = plot (minFitness);
% ylabel ("Min Fitness");
% title ("PSO Algorithm");
% waitfor(p);

Theta1 = reshape(best_global(1:hidden_layer * (input_layer + 1)), ...
                 hidden_layer, (input_layer + 1));

Theta2 = reshape(best_global((1 + (hidden_layer * (input_layer + 1))):end), ...
                 num_labels, (hidden_layer + 1));

data = dlmread('cancer_testing.data', ',', 0, 0);  % Read only numeric data
testing_data = data(:, 2:10); %This is the testing data
testing_labels = data(:, 11:11); %This is the testing data

result = predict(Theta1, Theta2, testing_data);
%disp("Predicted Results: ");
%disp(result);
%disp("Actual Results: ");
%disp(testing_labels);
training_acc = mean(double(result == testing_labels(1:length(result), 1))) * 100;
fprintf('Training Accuracy: %.2f%%\n', training_acc);

% Log constants, accuracy, and execution date
current_date = datestr(now, 'yyyy-mm-dd HH:MM:SS');
fid1 = fopen('logs.txt', 'a');
fprintf(fid1, "========================================\n");
fprintf(fid1, "Execution Date: %s\n", current_date);
fprintf(fid1, "========================================\n");
fprintf(fid1, "Constants Used:\n");
fprintf(fid1, "Input Layer: %d\n", input_layer);
fprintf(fid1, "Hidden Layer: %d\n", hidden_layer);
fprintf(fid1, "Number of Labels: %d\n", num_labels);
fprintf(fid1, "Max Generations: %d\n", MAX_GENERATIONS);
fprintf(fid1, "Total Population: %d\n", TOTAL_POPULATION);
fprintf(fid1, "Cognitive Coefficient (c1): %.2f\n", c1);
fprintf(fid1, "Social Coefficient (c2): %.2f\n", c2);
fprintf(fid1, "Max Inertia Weight (wmax): %.2f\n", wmax);
fprintf(fid1, "Min Inertia Weight (wmin): %.2f\n", wmin);
fprintf(fid1, "Max Velocity: %.2f \n", V_max);
fprintf(fid1,  "Best Global Fitness: %.7f\n", global_fitness);
fprintf(fid1, "========================================\n");
fprintf(fid1, "Training Accuracy: %.2f%%\n", training_acc);
fprintf(fid1, "========================================\n\n");
fclose(fid1);