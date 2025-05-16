function position = updatePosition(position, velocity)
  % Inputs:
  % position, velocity: 1xN cell arrays of 1xD vectors
  % Output:
  % new_position: 1xN cell array of updated positions

  N = numel(position);

  for i = 1:N
    position{i} = position{i} + velocity{i};
  end
end