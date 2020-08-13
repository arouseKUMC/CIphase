function [std_values, phase_pts] = phase_moving_std(phase, values, n_pts, varargin)
%[mean_values, phase_pts] = phase_moving_average(phase, values, n_pts)
%[mean_values, phase_pts] = phase_moving_average(phase, values, n_pts, 'wrap', false) 
%wrap option if true (default) 
%Originally tried to use fit with lowess (locally weighted smoothing) but was slow and not really necessary

p = inputParser;
addParameter(p,'wrap',true)
addParameter(p,'phaseRange', []);

parse(p, varargin{:})
if p.Results.wrap
    phase = wrapTo2Pi(phase);
    if isempty( p.Results.phaseRange)
        phase_pts = (2*pi/n_pts):(2*pi/n_pts):(2*pi);
    else
        phase_pts = p.Results.phaseRange(1):(2*pi/n_pts):p.Results.phaseRange(2);
    end
else
    if isempty( p.Results.phaseRange)
        phase_pts = ((2*pi)*floor(min(phase)/(2*pi))):(2*pi/n_pts):((2*pi)*ceil(max(phase)/(2*pi)));
    else
        phase_pts = p.Results.phaseRange(1):(2*pi/n_pts):p.Results.phaseRange(2);
    end
end
std_values = zeros(length(phase_pts),size(values,2));

for k = 1:length(phase_pts)
    if k == 1
        curr_indexes = phase>0 & phase<phase_pts(2);
    elseif k == length(phase_pts)
        curr_indexes = (phase>phase_pts(end-1) | phase<phase_pts(1)); 
    else
        curr_indexes = phase>phase_pts(k-1) & phase<phase_pts(k+1);
    end
    std_values(k,:) = std(values(curr_indexes,:));
end
    

