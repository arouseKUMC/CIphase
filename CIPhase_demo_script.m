%Use to add CIPhase to your Matlab path or add permanently
addpath('.\CIPhase\')
%Load example data
load('.\example_data\example_CIPhase_data.mat')


%Square root of Firing rates
sqrtSpikeFiringRates = sqrt(SpikeFiringRates);

%sqrtSpikeFiringRates is trial x time x neuron
n_tr = size(sqrtSpikeFiringRates,1); 

n_pc_for_jpc = 6;  %Dimensions of PCA on GT activity to submit to jPCA
jPC_filt_cutoffs = [0.5,5];    %Bandpass filter cutoffs, Hz
num_jpc_iterations = 3;  %Iterative number of times to re-align data
num_jpc_phase_pts = 100; %Number of phase bins from 0 to 2pi to divide data into for averaging
    
%Call find_CIDims to identify the neural dimensions with the most
%rotational condition-independent neural activity
CIDims = find_CIDims(sqrtSpikeFiringRates, dataMask_peakSpeed, 'trialNum', trial_ids_peakSpeed, ...
    'n_pc_for_jpc', n_pc_for_jpc, 'samp_rate', samp_rate, 'jPC_filt_cutoffs', jPC_filt_cutoffs, 'num_iterations', num_jpc_iterations, 'num_phase_pts', num_jpc_phase_pts);


nan_mask = double(~isnan(CursorSpeed));
nan_mask(nan_mask==0) = NaN;
sqrtSpikeFiringRates = sqrtSpikeFiringRates.*repmat(nan_mask, [1,1,size( sqrtSpikeFiringRates,3)]); 
TrialFiringRate_peakSpeed = getMaskedData(sqrtSpikeFiringRates, dataMask_peakSpeed, trial_ids_peakSpeed);

%Project firing rates into the identified Condition-Independent dimensions
CIDimsTrialFiringRate = reshape(reshape(sqrtSpikeFiringRates, [], size( sqrtSpikeFiringRates,3))*CIDims, n_tr, [], n_pc_for_jpc );
%Pull out data aligned to peak speed
CIDimsTrialFiringRate_peakSpeed = getMaskedData(CIDimsTrialFiringRate, dataMask_peakSpeed, trial_ids_peakSpeed);
global_jPCTrialFiringRate = nanmean(nanmean(CIDimsTrialFiringRate_peakSpeed,2),1);
%Subtract global mean for data away
CIDimsTrialFiringRate = bsxfun(@minus, CIDimsTrialFiringRate,global_jPCTrialFiringRate);  
CIDimsTrialFiringRate_peakSpeed = getMaskedData(CIDimsTrialFiringRate, dataMask_peakSpeed, trial_ids_peakSpeed);




[b,a] = butter(1, jPC_filt_cutoffs/(samp_rate/2), 'bandpass');

tmp = CIDimsTrialFiringRate;
    for tr = 1:n_tr
        tmp(tr,any(isnan(tmp(tr,:,:)),3),:) = repmat(tmp(tr,find(~any(isnan(tmp(tr,:,:)),3),1,'last'),:),  [1, sum(any(isnan(tmp(tr,:,:)),3)),1]);  %For filtfilt we make all the nan values at end equal to the last value
    end
    filt_jPCTrialFiringRate = permute(filtfilt(b,a, permute(tmp,[2,1,3])),[2,1,3]);
    filt_jPCTrialFiringRate(isnan(CIDimsTrialFiringRate)) = NaN;
    CI_dim_Phase_val = NaN(size(CIDimsTrialFiringRate(:,:,1:2)));
    for tr = 1:n_tr
        for dim = 1:2
        tmp = hilbert(filt_jPCTrialFiringRate(tr,~isnan(filt_jPCTrialFiringRate(tr,:,dim)),dim));  %Perform Hilbert transform on filtered jPC1
        CI_dim_Phase_val(tr,~isnan(filt_jPCTrialFiringRate(tr,:,1)),dim) = angle(tmp);                     %Get phase of Hilbert transform
        end
    end
CI_Phase_val = circ_mean(cat(3,CI_dim_Phase_val(:,:,1),wrapToPi(CI_dim_Phase_val(:,:,2)+pi/2)),3);
    


    