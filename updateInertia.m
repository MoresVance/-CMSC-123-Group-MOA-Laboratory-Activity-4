function inertia = updateInertia(personal_best, global_best, wmax, wmin, current_i, max_i)
  % Inputs: 
  % personal_best: scalar, fitness of the personal best position
  % global_best: scalar, fitness of the global best position
  % wmax: maximum inertia weight
  % wmin: minimum inertia weight
  % current_i: current iteration number
  % max_i: maximum number of iterations

  % Equation (4) -  Crucially, limit mu!
  mu_raw = (personal_best - global_best) / personal_best;
  mu = max(min(mu_raw, 30), -30);  % Limit mu to [-n, n] 

  % Equation (5)
  delta = wmax - ((wmax - wmin) * current_i / max_i);

  % Equation (3)
  w_unbounded = mu * tanh(delta);
  %  Scale w to the range [Wmin, Wmax]
  w = wmin + ((wmax - wmin) * ((w_unbounded + 1) / 2));
  %  Further ensure w is within bounds
  inertia = max(min(w, wmax), wmin);

end