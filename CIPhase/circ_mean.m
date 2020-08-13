
function mean_r = circ_mean(alpha,dim)

% Compute sum of imaginary vectors
r = sum(exp(1i*alpha),dim);

% Calcuate angle imaginary sum
mean_r = angle(r);