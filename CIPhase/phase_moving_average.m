function [mean_values, phase_pts, mean_ci_values] = phase_moving_average(phase, values, n_pts, varargin)
%[mean_values, phase_pts] = phase_moving_average(phase, values, n_pts)
%[mean_values, phase_pts] = phase_moving_average(phase, values, n_pts, 'wrap', false) 
%wrap option if true (default) 
%[mean_values, phase_pts, mean_ci_values] = phase_moving_average(phase, values, n_pts, 'CI_Limit', 95) 
% Specify CI_Limit as the percentile for confidence interval to return mean confidence intervals, typically 95
%[mean_values, phase_pts, mean_ci_values] = phase_moving_average(phase, values, n_pts, 'CI_Limit', 95, 'CI_Iterations', 1000) 
%
%Originally tried to use fit with lowess (locally weighted smoothing) but was slow and not really necessary

p = inputParser;
addParameter(p,'wrap',true)
addParameter(p,'phaseRange', []);
addParameter(p,'CI_Limit', []);
addParameter(p,'CI_Iterations', 1000);

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
mean_values = zeros(length(phase_pts),size(values,2));
if ~isempty(p.Results.CI_Limit)
    mean_ci_values = zeros(length(phase_pts),size(values,2),2);
end
for k = 1:length(phase_pts)
    if k == 1 && p.Results.wrap                           %Ranges are centered on a phase point and go one point less to one point more
        curr_indexes = phase>0 & phase<phase_pts(2);
    elseif k == 1 && ~p.Results.wrap                           
        curr_indexes = phase>( phase_pts(1)-(2*pi/n_pts) ) & phase<phase_pts(2);
    elseif k == length(phase_pts) && p.Results.wrap
        curr_indexes = (phase>phase_pts(end-1) | phase<phase_pts(1));
    elseif k == length(phase_pts) && ~p.Results.wrap                           
       curr_indexes = phase>phase_pts(end-1) & phase<( phase_pts(end)+(2*pi/n_pts) );
    else
        curr_indexes = phase>phase_pts(k-1) & phase<phase_pts(k+1);
    end
    mean_values(k,:) = mean(values(curr_indexes,:));
    if ~isempty(p.Results.CI_Limit)
        curr_indexes = find(curr_indexes);
        if ~isempty(curr_indexes)
            curr_mean_values = zeros(1,size(values,2), p.Results.CI_Iterations);
            for ii = 1:p.Results.CI_Iterations
                curr_mean_values(1,:,ii) = mean(values( curr_indexes(randsample(length(curr_indexes),length(curr_indexes),true)),:));
            end
            mean_ci_values(k,:,:) = prctile(curr_mean_values, [(100-p.Results.CI_Limit)/2, 100-(100-p.Results.CI_Limit)/2], 3);
        else
            mean_ci_values(k,:,:) = NaN;
        end
    end
end
    

