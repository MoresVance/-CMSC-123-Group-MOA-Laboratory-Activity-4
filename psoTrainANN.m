% This file implements the Particle Swarm Optimization algorithm for training the ANN.
function optimal_weights = psoTrainANN(X, y, input_layer, hidden_layer, num_labels, MAX_ITERATIONS, SWARM_SIZE)

    % Initialize the swarm
    swarm = cell(SWARM_SIZE, 1);
    velocities = cell(SWARM_SIZE, 1);
    personal_best = cell(SWARM_SIZE, 1);
    personal_best_fitness = inf(SWARM_SIZE, 1);
    global_best = [];
    global_best_fitness = inf;

    % Generate initial particles
    for i = 1:SWARM_SIZE
        swarm{i} = rand(hidden_layer * (input_layer + 1) + num_labels * (hidden_layer + 1), 1) * 2 - 1; % Random weights
        velocities{i} = zeros(size(swarm{i})); % Initialize velocities
    end

    % PSO main loop
    for iter = 1:MAX_ITERATIONS
        for i = 1:SWARM_SIZE
            % Evaluate fitness
            fitness = nnCostFunction(swarm{i}, input_layer, hidden_layer, num_labels, X, y, 1);
            
            % Update personal best
            if fitness < personal_best_fitness(i)
                personal_best_fitness(i) = fitness;
                personal_best{i} = swarm{i};
            end
            
            % Update global best
            if fitness < global_best_fitness
                global_best_fitness = fitness;
                global_best = swarm{i};
            end
        end
        
        % Update velocities and positions
        for i = 1:SWARM_SIZE
            r1 = rand(size(swarm{i}));
            r2 = rand(size(swarm{i}));
            velocities{i} = 0.5 * velocities{i} + ...
                            2 * r1 .* (personal_best{i} - swarm{i}) + ...
                            2 * r2 .* (global_best - swarm{i});
            swarm{i} = swarm{i} + velocities{i};
        end
        
        fprintf('Iteration %d: Global Best Fitness = %.4f\n', iter, global_best_fitness);
    end

    optimal_weights = global_best; % Return the optimal weights
end