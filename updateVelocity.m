function velocity = updateVelocity(current_velocity, position, personal_best, global_best, c1, c2, current_i, max_i, wmin, wmax, personal_best_fitness, global_best_fitness)
  % Inputs:
  % current_velocity, position, personal_best: 1xN cell arrays of 1xD vectors
  % global_best: 1xD vector (not a cell)
  % w: inertia weight
  % c1, c2: cognitive and social coefficients

  N = numel(position); % Number of particles

  for i = 1:N
    % Generate random vectors of the same size as the particle
    D = numel(position{i});

    % Velocity update rule (element-wise operations)
    w = updateInertia(personal_best_fitness, global_best_fitness, wmax, wmin, current_i, max_i);
    inertia    = w  * current_velocity{i};
    cognitive  = c1 * (rand() * 2 * 0.12 - 0.12) * (personal_best{i} - position{i});
    social     = c2 * (rand() * 2 * 0.12 - 0.12) * (global_best - position{i});

    velocity{i} = inertia + cognitive + social;
  end
end